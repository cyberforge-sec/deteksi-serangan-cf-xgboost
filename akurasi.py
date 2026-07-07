"""
Hybrid Certainty Factor–XGBoost for Cyberattack Detection
=========================================================

Implementation of:

    Aprianto, A. D., Maharrani, R. H., Auliya, I. C. R., & Alifiah, V. R. (2026).
    A Hybrid Certainty Factor–XGBoost Approach for Cyberattack Detection Using
    the TON_IoT Dataset. Journal of Information Systems and Informatics, 8(2), 1519.
    https://doi.org/10.63158/journalisi.v8i2.1519

Combines rule-based reasoning (Certainty Factor) with gradient boosting (XGBoost)
via meta-learning for network intrusion detection on the TON_IoT dataset.

Usage:
    python akurasi.py --dataset path/to/data.csv [--test-size 0.2] [--output output/]

Author: cyberforge-sec
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from xgboost import XGBClassifier, plot_importance

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plotting setup
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_palette("husl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_ATTACK_VALUES = {"attack", "malicious", "1", "true", "anomaly"}

COLUMNS_TO_DROP = [
    "src_ip", "dst_ip", "dns_query", "ssl_subject", "ssl_issuer",
    "http_user_agent", "weird_name", "weird_addl", "type", "dns_qclass",
]

NUMERIC_COLUMNS = ["duration", "src_bytes", "dst_bytes", "dns_qtype"]

CATEGORICAL_COLUMNS = [
    "conn_state", "proto", "service", "http_status_code",
    "weird_notice", "http_method", "ssl_resumed", "http_uri",
]

# CF engine defaults
CF_MIN_SUPPORT = 0.05
CF_MIN_CONFIDENCE = 0.6
CF_MB_MD_THRESHOLD = 0.8
CF_MAX_ACTIVE_RULES = 10

# XGBoost defaults
XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 1,
}

HYBRID_PARAMS: Dict[str, Any] = {
    "n_estimators": 50,
    "max_depth": 2,
    "learning_rate": 0.05,
    "random_state": 42,
    "eval_metric": "logloss",
    "subsample": 0.7,
    "colsample_bytree": 0.7,
}

OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CFRule:
    """A single Certainty Factor rule extracted from data."""

    column: str
    operator: str          # "==" | "<" | ">"
    value: Any
    mb: float              # measure of belief
    md: float              # measure of disbelief

    @classmethod
    def from_string(cls, column: str, kondisi: str, mb: float, md: float) -> "CFRule":
        """Parse a condition string and create a rule."""
        if " == " in kondisi:
            parts = kondisi.split(" == ")
            val = parts[1].strip().strip("'\"")
            return cls(column=column, operator="==", value=val, mb=mb, md=md)
        elif " < " in kondisi:
            parts = kondisi.split(" < ")
            return cls(column=column, operator="<", value=float(parts[1].strip()), mb=mb, md=md)
        elif " > " in kondisi:
            parts = kondisi.split(" > ")
            return cls(column=column, operator=">", value=float(parts[1].strip()), mb=mb, md=md)
        else:
            raise ValueError(f"Unrecognized condition format: {kondisi}")

    def evaluate(self, row: Dict[str, Any]) -> bool:
        """Test if this rule's condition is satisfied by a row dict."""
        if self.column not in row or pd.isna(row[self.column]):
            return False

        nilai = row[self.column]

        try:
            if self.operator == "==":
                return str(nilai).strip().lower() == str(self.value).strip().lower()
            elif self.operator == "<":
                return pd.to_numeric(nilai, errors="coerce") < self.value
            elif self.operator == ">":
                return pd.to_numeric(nilai, errors="coerce") > self.value
        except (ValueError, TypeError):
            return False

        return False


# ---------------------------------------------------------------------------
# CF Engine
# ---------------------------------------------------------------------------

class CFEngine:
    """
    Certainty Factor inference engine.

    Computes CF scores from a set of rules against data rows.
    Thread-safe: rules are read-only after construction.
    """

    def __init__(self, rules: List[CFRule]):
        self.rules = rules

    @classmethod
    def build_from_data(
        cls,
        df: pd.DataFrame,
        label_col: str = "label_binary",
        min_support: float = CF_MIN_SUPPORT,
        min_confidence: float = CF_MIN_CONFIDENCE,
    ) -> "CFEngine":
        """Automatically extract CF rules from labelled training data."""
        rules: List[CFRule] = []
        total = len(df)
        attack = df[df[label_col] == 1]
        normal = df[df[label_col] == 0]

        if len(attack) == 0 or len(normal) == 0:
            log.warning("Only one class present — cannot build discriminative rules")
            return cls(rules=[])

        def _mb_md_cat(feature: str, value: Any) -> Tuple[float, float]:
            """MB/MD for categorical equality (feature == value)."""
            hit_feature = len(df[df[feature] == value])
            hit_attack = len(attack[attack[feature] == value])
            hit_normal = len(normal[normal[feature] == value])

            if hit_feature == 0:
                return 0.0, 0.0

            p_attack = hit_attack / len(attack)
            p_normal = hit_normal / len(normal)

            mb = max(0.0, (p_attack - p_normal)) / (p_attack + p_normal + 1e-9)
            md = max(0.0, (p_normal - p_attack)) / (p_attack + p_normal + 1e-9)

            return min(mb, 1.0), min(md, 1.0)

        def _mb_md_num(feature: str, op: str, threshold: float) -> Tuple[float, float]:
            """MB/MD for numeric range (feature < threshold or feature > threshold)."""
            if op == "<":
                mask = df[feature] < threshold
            else:
                mask = df[feature] > threshold

            hit_attack = mask[attack.index].sum()
            hit_normal = mask[normal.index].sum()

            p_attack = hit_attack / len(attack)
            p_normal = hit_normal / len(normal)

            mb = max(0.0, (p_attack - p_normal)) / (p_attack + p_normal + 1e-9)
            md = max(0.0, (p_normal - p_attack)) / (p_attack + p_normal + 1e-9)

            return min(mb, 1.0), min(md, 1.0)

        # Categorical rules
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
            for value, count in df[col].value_counts().items():
                if count / total < min_support:
                    continue
                mb, md = _mb_md_cat(col, value)
                confidence = mb / (mb + md + 1e-9)
                if confidence >= min_confidence and (mb > CF_MB_MD_THRESHOLD or md > CF_MB_MD_THRESHOLD):
                    rules.append(CFRule.from_string(col, f"{col} == '{value}'", mb, md))

        # Numeric rules (quantile-based)
        for col in ["duration", "src_bytes", "dst_bytes", "dns_qtype", "bytes_ratio", "packet_rate"]:
            if col not in df.columns:
                continue
            q_low, q_high = df[col].quantile(0.05), df[col].quantile(0.95)

            mb_low, md_low = _mb_md_num(col, "<", q_low)
            if max(mb_low, md_low) > 0.1:
                rules.append(CFRule.from_string(col, f"{col} < {q_low}", md_low, mb_low))

            mb_high, md_high = _mb_md_num(col, ">", q_high)
            if max(mb_high, md_high) > 0.1:
                rules.append(CFRule.from_string(col, f"{col} > {q_high}", mb_high, md_high))

        log.info("Built %d CF rules from training data", len(rules))
        return cls(rules=rules)

    def score_row(self, row: Dict[str, Any]) -> float:
        """Compute combined CF score for a single row (0..1)."""
        if not self.rules:
            return 0.0

        total_cf = 0.0
        active = 0

        for rule in self.rules:
            if not rule.evaluate(row):
                continue

            cf = rule.mb - rule.md
            if abs(total_cf) < 1e-6:
                total_cf = cf
            else:
                total_cf += cf * (1 - abs(total_cf))
            active += 1

        if active > 0:
            total_cf /= min(active, CF_MAX_ACTIVE_RULES)

        # Sigmoid normalization
        return float(1.0 / (1.0 + np.exp(-5.0 * total_cf)))

    def score_bulk(self, df: pd.DataFrame, n_jobs: Optional[int] = None) -> np.ndarray:
        """
        Score all rows, optionally using multiprocessing.

        Falls back to sequential on error (e.g., pickling issues).
        """
        if not self.rules:
            return np.zeros(len(df))

        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 1)

        try:
            return self._score_parallel(df, n_jobs)
        except Exception as exc:
            log.warning("Parallel scoring failed (%s), falling back to sequential", exc)
            return self._score_sequential(df)

    def _score_sequential(self, df: pd.DataFrame) -> np.ndarray:
        """Score rows one by one."""
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CF sequential"):
            scores.append(self.score_row(row.to_dict()))
        return np.array(scores)

    def _score_parallel(self, df: pd.DataFrame, n_jobs: int) -> np.ndarray:
        """Score rows in parallel chunks."""
        chunk_size = max(1, len(df) // (n_jobs * 10))
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        log.info("Scoring %d rows with %d workers (%d chunks)", len(df), n_jobs, len(chunks))

        # Serialise rules once — avoids pickling the whole engine
        rule_dicts = [{"col": r.column, "op": r.operator, "val": r.value, "mb": r.mb, "md": r.md}
                       for r in self.rules]

        scores: List[float] = []
        with Pool(n_jobs) as pool:
            worker = partial(_score_chunk, rule_dicts)
            for result in tqdm(pool.imap_unordered(worker, chunks), total=len(chunks), desc="CF parallel"):
                scores.extend(result)

        return np.array(scores)


def _score_chunk(rule_dicts: List[Dict[str, Any]], chunk: pd.DataFrame) -> List[float]:
    """Worker function for multiprocessing — module-level so it's pickleable."""
    rules = [
        CFRule(column=r["col"], operator=r["op"], value=r["val"], mb=r["mb"], md=r["md"])
        for r in rule_dicts
    ]
    engine = CFEngine(rules)
    return [engine.score_row(row.to_dict()) for _, row in chunk.iterrows()]


# ---------------------------------------------------------------------------
# Adaptive Threshold (KMeans-based)
# ---------------------------------------------------------------------------

@dataclass
class RiskThresholds:
    low: float = 0.3
    medium: float = 0.7

    @classmethod
    def from_scores(cls, scores: np.ndarray) -> "RiskThresholds":
        """Determine thresholds adaptively using KMeans clustering."""
        unique = np.unique(scores)
        if len(unique) < 3:
            log.warning("%d unique CF values — using default thresholds", len(unique))
            return cls()

        n_clusters = min(3, len(unique))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(scores.reshape(-1, 1))
        centers = sorted(kmeans.cluster_centers_.flatten())

        if len(centers) >= 3:
            low = centers[0] + (centers[1] - centers[0]) * 0.4
            medium = centers[1] + (centers[2] - centers[1]) * 0.6
        elif len(centers) == 2:
            low = centers[0] + (centers[1] - centers[0]) * 0.3
            medium = centers[0] + (centers[1] - centers[0]) * 0.7
        else:
            return cls()

        result = cls(low=float(low), medium=float(medium))
        log.info("Adaptive thresholds — low=%.3f, medium=%.3f", result.low, result.medium)
        return result

    def classify(self, score: float, buffer: float = 0.05) -> str:
        """Map a CF score to a risk level string."""
        if score < self.low - buffer:
            return "rendah"
        elif self.low + buffer <= score < self.medium - buffer:
            return "sedang"
        elif score >= self.medium + buffer:
            return "tinggi"
        else:
            return "rendah" if score < self.medium else "sedang"


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> Optional[pd.DataFrame]:
    """Load a CSV dataset with basic validation."""
    if not os.path.exists(path):
        log.error("File not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        log.info("Loaded %s — %d rows, %d columns", path, len(df), len(df.columns))
        return df
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        return None


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform raw network data.

    * Drops high-cardinality / uninformative columns
    * Log-normalises large numeric values
    * Encodes categorical strings
    * Engineers interaction features
    """
    log.info("Preprocessing %d rows...", len(df))

    # Drop unwanted columns
    existing = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        log.info("Dropped %d uninformative columns", len(existing))

    # Numeric columns
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if df[col].max() > 1e6:
            df[col] = np.log1p(df[col])
            log.info("Log-transformed '%s' (skewed)", col)
        df[col] = df[col].fillna(0)

    # Categorical clean-up
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.lower().replace("nan", "unknown")

    # Feature engineering
    if "src_bytes" in df.columns and "dst_bytes" in df.columns:
        df["bytes_ratio"] = np.log1p(df["src_bytes"] / (df["dst_bytes"] + 1))
    if all(c in df.columns for c in ["duration", "src_bytes", "dst_bytes"]):
        df["packet_rate"] = df["duration"] / (df["src_bytes"] + df["dst_bytes"] + 1)

    log.info("Preprocessing complete — %d columns", len(df.columns))
    return df


def encode_categoricals(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns.

    Parameters
    ----------
    df : DataFrame
        Data to encode.
    encoders : dict or None
        Pre-fitted encoders to reuse (when *fit* is False).
    fit : bool
        If True, fit new encoders; otherwise transform using existing ones.

    Returns
    -------
    (DataFrame, dict of encoders)
    """
    if encoders is None:
        encoders = {}

    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    df = df.copy()

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                continue
            # Handle unknown categories gracefully
            df[col] = df[col].astype(str).map(
                lambda x: x if x in le.classes_ else "unknown"
            )
            # If "unknown" wasn't in training classes, add it
            if "unknown" not in le.classes_:
                classes = list(le.classes_) + ["unknown"]
                le.classes_ = np.array(classes)
            df[col] = le.transform(df[col])

    return df, encoders


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_cf_distribution(
    df: pd.DataFrame,
    thresholds: RiskThresholds,
    save_path: str = "output/cf_distribution.png",
) -> None:
    """KDE of CF scores per risk level."""
    plt.figure(figsize=(10, 6))
    for level, color in [("rendah", "green"), ("sedang", "orange"), ("tinggi", "red")]:
        subset = df[df["tingkat_risiko"] == level]
        if not subset.empty:
            sns.kdeplot(subset["cf_score"], label=level.capitalize(), color=color, fill=True)
    plt.axvline(thresholds.low, color="blue", linestyle="--", label="Low boundary")
    plt.axvline(thresholds.medium, color="purple", linestyle="--", label="Medium boundary")
    plt.title("CF Score Distribution by Risk Level")
    plt.xlabel("CF Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info("Saved %s", save_path)


def plot_risk_distribution(df: pd.DataFrame, save_path: str = "output/risk_distribution.png") -> None:
    """Pie chart of predicted risk levels."""
    dist = df["tingkat_risiko"].value_counts()
    colors = {"rendah": "green", "sedang": "orange", "tinggi": "red"}
    plt.figure(figsize=(8, 6))
    plt.pie(
        dist,
        labels=dist.index,
        colors=[colors.get(lvl, "gray") for lvl in dist.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    plt.title("Risk Level Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info("Saved %s", save_path)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    save_path: str,
) -> None:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info("Saved %s", save_path)


def plot_roc_curves(
    df: pd.DataFrame,
    save_path: str = "output/roc_curves.png",
) -> None:
    """Overlaid ROC curves for all available models."""
    plt.figure(figsize=(10, 8))
    for key, label, color in [
        ("cf_score", "CF", "blue"),
        ("prob_xgb", "XGBoost", "green"),
        ("prob_hybrid", "Hybrid", "red"),
    ]:
        if key in df.columns and "label_binary" in df.columns:
            fpr, tpr, _ = roc_curve(df["label_binary"], df[key])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})", linewidth=2, color=color)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info("Saved %s", save_path)


def plot_score_comparison(df: pd.DataFrame, save_path: str = "output/score_comparison.png") -> None:
    """Scatter plot of XGBoost probability vs CF score."""
    if not all(c in df.columns for c in ["prob_xgb", "cf_score", "label_binary"]):
        return
    sample = df.sample(min(1000, len(df)))
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="prob_xgb", y="cf_score", hue="label_binary",
        data=sample, alpha=0.6, palette={0: "green", 1: "red"},
    )
    plt.title("XGBoost Probability vs CF Score")
    plt.xlabel("XGBoost Probability")
    plt.ylabel("CF Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info("Saved %s", save_path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: pd.Series,
    y_score: pd.Series,
    model_name: str,
) -> Dict[str, Any]:
    """Compute standard classification metrics."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5

    y_pred = (y_score >= best_threshold).astype(int)

    return {
        "model": model_name,
        "threshold": float(best_threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "y_pred": y_pred,
    }


def print_metrics(result: Dict[str, Any]) -> None:
    """Pretty-print evaluation metrics."""
    log.info("── %s ──", result["model"])
    log.info("  Optimal threshold:   %.4f", result["threshold"])
    log.info("  Accuracy:            %.2f%%", result["accuracy"] * 100)
    log.info("  Balanced Accuracy:   %.2f%%", result["balanced_accuracy"] * 100)
    log.info("  MCC:                 %.4f", result["mcc"])
    cm = result["confusion_matrix"]
    log.info("  Confusion Matrix:")
    log.info("    TN=%d  FP=%d", cm[0][0], cm[0][1])
    log.info("    FN=%d  TP=%d", cm[1][0], cm[1][1])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Network Anomaly Detection — CF + XGBoost + Hybrid"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="train_test_network.csv",
        help="Path to the full dataset CSV (will be split into train/test). Default: train_test_network.csv",
    )
    parser.add_argument(
        "--test-size", "-t",
        type=float,
        default=0.2,
        help="Fraction held out as test set (default: 0.2). The rest is used for training.",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for plots and results (default: output/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 50)
    log.info("Network Anomaly Detection Pipeline")
    log.info("  Dataset:   %s", args.dataset)
    log.info("  Test size: %.2f", args.test_size)
    log.info("  Output:    %s/", output_dir)
    log.info("=" * 50)

    # 1. Load & preprocess ----------------------------------------------------
    df = load_dataset(args.dataset)
    if df is None:
        sys.exit(1)

    df = preprocess(df)
    if "label" not in df.columns:
        log.error("Column 'label' not found — cannot continue")
        sys.exit(1)

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["label_binary"] = df["label"].apply(lambda x: 1 if x in LABEL_ATTACK_VALUES else 0)

    # Verify both classes exist
    if df["label_binary"].nunique() < 2:
        log.error("Dataset must contain both normal and attack samples")
        sys.exit(1)

    attack_pct = df["label_binary"].mean() * 100
    log.info("Class balance: %.1f%% normal, %.1f%% attack", 100 - attack_pct, attack_pct)

    # 2. Train/Test split ----------------------------------------------------
    # IMPORTANT: Split FIRST to prevent data leakage across the pipeline.
    drop_cols = ["label", "label_binary", "cf_score", "tingkat_risiko"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["label_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y,
    )

    # Reconstruct full DataFrames for convenience
    train_idx = X_train.index
    test_idx = X_test.index
    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()

    log.info("Train: %d rows | Test: %d rows", len(df_train), len(df_test))

    # 3. Encode categoricals --------------------------------------------------
    df_train, encoders = encode_categoricals(df_train, fit=True)
    df_test, _ = encode_categoricals(df_test, encoders=encoders, fit=False)

    # 4. CF Engine — build from TRAIN only ------------------------------------
    cf_engine = CFEngine.build_from_data(df_train)
    rules_df = pd.DataFrame([
        {"kolom": r.column, "kondisi": f"{r.column} {r.operator} {r.value}",
         "mb": r.mb, "md": r.md}
        for r in cf_engine.rules
    ])
    if not rules_df.empty:
        rules_df.to_csv(output_dir / "basis_pengetahuan_otomatis.csv", index=False)
        log.info("Saved %d rules to basis_pengetahuan_otomatis.csv", len(rules_df))

    # 5. CF scoring (train + test) -------------------------------------------
    log.info("Scoring training set with CF...")
    df_train["cf_score"] = cf_engine.score_bulk(df_train)
    log.info("Scoring test set with CF...")
    df_test["cf_score"] = cf_engine.score_bulk(df_test)

    # Adaptive thresholds from TRAIN scores only
    thresholds = RiskThresholds.from_scores(df_train["cf_score"].values)
    df_train["tingkat_risiko"] = df_train["cf_score"].apply(thresholds.classify)
    df_test["tingkat_risiko"] = df_test["cf_score"].apply(thresholds.classify)

    # 6. XGBoost — train on TRAIN only --------------------------------------
    train_cols = [c for c in drop_cols if c in df_train.columns]
    X_train_xgb = df_train.drop(columns=train_cols, errors="ignore")

    xgb_model = XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train_xgb, y_train)

    # Predict on both splits
    X_test_xgb = df_test.drop(columns=[c for c in drop_cols + ["predicted_cf", "predicted_xgb", "predicted_hybrid"]
                                        if c in df_test.columns], errors="ignore")
    df_train["predicted_xgb"] = xgb_model.predict(X_train_xgb)
    df_train["prob_xgb"] = xgb_model.predict_proba(X_train_xgb)[:, 1]
    df_test["predicted_xgb"] = xgb_model.predict(X_test_xgb)
    df_test["prob_xgb"] = xgb_model.predict_proba(X_test_xgb)[:, 1]

    # 7. Hybrid model — train on TRAIN only ----------------------------------
    hybrid_features_train = np.column_stack([df_train["prob_xgb"], df_train["cf_score"]])
    hybrid_features_test = np.column_stack([df_test["prob_xgb"], df_test["cf_score"]])

    hybrid_model = XGBClassifier(**HYBRID_PARAMS)
    hybrid_model.fit(hybrid_features_train, y_train)

    df_train["predicted_hybrid"] = hybrid_model.predict(hybrid_features_train)
    df_train["prob_hybrid"] = hybrid_model.predict_proba(hybrid_features_train)[:, 1]
    df_test["predicted_hybrid"] = hybrid_model.predict(hybrid_features_test)
    df_test["prob_hybrid"] = hybrid_model.predict_proba(hybrid_features_test)[:, 1]

    # 8. Evaluate ALL models on the TEST set only ---------------------------
    log.info("\n" + "=" * 50)
    log.info("FINAL EVALUATION (test set — %.0f rows)", len(df_test))
    log.info("=" * 50)

    # CF
    log.info("")
    metrics_cf = evaluate_model(y_test, df_test["cf_score"], "Certainty Factor")
    print_metrics(metrics_cf)
    plot_confusion_matrix(y_test, metrics_cf["y_pred"], "CF", str(output_dir / "cf_confusion_matrix.png"))

    # XGBoost
    log.info("")
    metrics_xgb = evaluate_model(y_test, df_test["prob_xgb"], "XGBoost")
    print_metrics(metrics_xgb)
    plot_confusion_matrix(y_test, metrics_xgb["y_pred"], "XGBoost", str(output_dir / "xgb_confusion_matrix.png"))

    # Hybrid — always available, trained unconditionally above
    log.info("")
    metrics_h = evaluate_model(y_test, df_test["prob_hybrid"], "Hybrid")
    print_metrics(metrics_h)
    plot_confusion_matrix(y_test, metrics_h["y_pred"], "Hybrid", str(output_dir / "hybrid_confusion_matrix.png"))

    # Shared plots
    plot_cf_distribution(df_test, thresholds, str(output_dir / "cf_distribution.png"))
    plot_risk_distribution(df_test, str(output_dir / "risk_distribution.png"))
    plot_roc_curves(df_test, str(output_dir / "roc_curves.png"))
    plot_score_comparison(df_test, str(output_dir / "score_comparison.png"))

    # Feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_importance(xgb_model, importance_type="weight", ax=ax)
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close()
    log.info("Saved feature_importance.png")

    # 9. Save results ---------------------------------------------------------
    df_test.to_csv(output_dir / "hasil_prediksi_enhanced.csv", index=False)
    log.info("\nResults saved to %s/hasil_prediksi_enhanced.csv", output_dir)
    log.info("All charts saved to %s/", output_dir)
    log.info("=" * 50)
    log.info("Pipeline complete!")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
