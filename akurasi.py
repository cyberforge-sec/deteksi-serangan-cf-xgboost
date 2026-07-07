import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, confusion_matrix,
                           classification_report, roc_curve, auc,
                           precision_recall_curve, f1_score,
                           balanced_accuracy_score, matthews_corrcoef)
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, plot_importance
import argparse
import os

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup visualisasi
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_palette("husl")

# --- Module-level function for multiprocessing ---
# (must be at module level so multiprocessing can pickle it across workers)


def _proses_chunk(args):
    """Module-level helper for multiprocessing. Processes a chunk of rows."""
    chunk_dicts, df_gejala_dicts = args
    detektor = CFSkorCalculator(df_gejala_dicts)
    return [detektor.hitung_cf(row) for row in chunk_dicts]


class CFSkorCalculator:
    """Helper class that holds rule data for CF computation.

    This exists solely so we can pickle it across multiprocessing workers
    without carrying the entire SistemDeteksiAnomali instance.
    """

    def __init__(self, df_gejala_dicts):
        # Convert list of dicts back to DataFrame internally
        self.df_gejala = pd.DataFrame(df_gejala_dicts) if df_gejala_dicts else None

    def hitung_cf(self, row_dict):
        """Menghitung Certainty Factor untuk satu baris - tanpa eval()."""
        if self.df_gejala is None or len(self.df_gejala) == 0:
            return 0.0

        total_cf = 0.0
        gejala_aktif = 0

        for _, gejala in self.df_gejala.iterrows():
            try:
                kolom = gejala['kolom']
                kondisi_str = gejala['kondisi']
                mb = gejala['mb']
                md = gejala['md']

                if kolom not in row_dict or pd.isna(row_dict[kolom]):
                    continue

                nilai = row_dict[kolom]
                kondisi_terpenuhi = False

                # --- Safe condition evaluation (no eval()) ---
                if ' == ' in kondisi_str:
                    # Format: "kolom == 'value'"
                    parts = kondisi_str.split(' == ')
                    value_part = parts[1].strip().strip("'\"")
                    kondisi_terpenuhi = str(nilai).strip().lower() == value_part.lower()

                elif ' < ' in kondisi_str:
                    # Format: "kolom < threshold"
                    try:
                        parts = kondisi_str.split(' < ')
                        threshold = float(parts[1].strip())
                        kondisi_terpenuhi = pd.to_numeric(nilai, errors='coerce') < threshold
                    except (ValueError, TypeError):
                        kondisi_terpenuhi = False

                elif ' > ' in kondisi_str:
                    # Format: "kolom > threshold"
                    try:
                        parts = kondisi_str.split(' > ')
                        threshold = float(parts[1].strip())
                        kondisi_terpenuhi = pd.to_numeric(nilai, errors='coerce') > threshold
                    except (ValueError, TypeError):
                        kondisi_terpenuhi = False

                if kondisi_terpenuhi:
                    cf = mb - md
                    if abs(total_cf) < 0.001:
                        total_cf = cf
                    else:
                        total_cf += cf * (1 - abs(total_cf))
                    gejala_aktif += 1

            except Exception:
                continue

        # Batasi maksimal 10 aturan yang berpengaruh
        if gejala_aktif > 0:
            total_cf = total_cf / min(gejala_aktif, 10)

        # Gunakan fungsi sigmoid untuk normalisasi
        return 1 / (1 + np.exp(-5 * total_cf))


class SistemDeteksiAnomali:
    def __init__(self):
        self.df_gejala = None
        self.thresholds = {'rendah': 0.3, 'sedang': 0.7}
        self.xgb_model = None
        self.hybrid_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fitur_dihapus = []

    def muat_data(self, file_data):
        """Memuat data dan melakukan preprocessing"""
        print(f"📥 Memuat dataset dari {file_data}...")

        if not os.path.exists(file_data):
            print(f"❌ File tidak ditemukan: {file_data}")
            return None

        try:
            df = pd.read_csv(file_data)
            print(f"✅ Dataset dimuat ({len(df)} baris, {len(df.columns)} kolom)")
            return df
        except Exception as e:
            print(f"❌ Gagal memuat dataset: {str(e)}")
            return None

    def praproses_data(self, df):
        """Melakukan preprocessing data"""
        print("\n🧹 Memulai preprocessing data...")

        # Drop columns yang tidak digunakan (pertahankan 'label')
        cols_to_drop = ['src_ip', 'dst_ip', 'dns_query', 'ssl_subject', 'ssl_issuer',
                       'http_user_agent', 'weird_name', 'weird_addl', 'type',
                       'dns_qclass']
        cols_existing = [col for col in cols_to_drop if col in df.columns]
        if cols_existing:
            df = df.drop(cols_existing, axis=1)
            print(f"  - Menghapus {len(cols_existing)} kolom yang tidak digunakan")

        # Proses kolom numerik
        kolom_numerik = ['duration', 'src_bytes', 'dst_bytes', 'dns_qtype']
        for col in kolom_numerik:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

                if df[col].max() > 1e6:
                    df[col] = np.log1p(df[col])
                    print(f"  - Log-normalisasi {col}")

                df[col] = df[col].fillna(0)

        print(f"  - {len([c for c in kolom_numerik if c in df.columns])} kolom numerik diproses")

        # Proses kolom kategorikal
        kolom_kategorikal = ['conn_state', 'proto', 'service', 'http_status_code',
                           'weird_notice', 'http_method', 'ssl_resumed', 'http_uri']
        for col in kolom_kategorikal:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].replace('nan', 'unknown')

        print(f"  - {len([c for c in kolom_kategorikal if c in df.columns])} kolom kategorikal diproses")

        # Tambahkan fitur baru
        if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
            df['bytes_ratio'] = np.log1p(df['src_bytes'] / (df['dst_bytes'] + 1))
        if 'duration' in df.columns and 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
            df['packet_rate'] = df['duration'] / (df['src_bytes'] + df['dst_bytes'] + 1)
        print("  - Fitur baru: bytes_ratio, packet_rate")

        return df

    def bangun_basis_pengetahuan(self, df, min_support=0.05, min_confidence=0.6):
        """Membangun basis pengetahuan secara otomatis dari data"""
        print("\n🔍 Membangun basis pengetahuan gejala secara otomatis...")
        basis_pengetahuan = []

        if 'label_binary' not in df.columns:
            if 'label' not in df.columns:
                print("❌ Kolom 'label' tidak ditemukan")
                return

            df['label_binary'] = df['label'].apply(
                lambda x: 1 if str(x).lower() in ['attack', 'malicious', '1', 'true'] else 0)

        total_records = len(df)
        attack_records = df[df['label_binary'] == 1]
        normal_records = df[df['label_binary'] == 0]

        if len(attack_records) == 0 or len(normal_records) == 0:
            print("⚠️ Data hanya memiliki satu kelas — tidak bisa membangun aturan diskriminatif")
            self.df_gejala = pd.DataFrame()
            return

        def hitung_mb_md(feature, value):
            total_feature = len(df[df[feature] == value])
            attack_feature = len(attack_records[attack_records[feature] == value])
            normal_feature = len(normal_records[normal_records[feature] == value])

            if total_feature == 0:
                return 0.0, 0.0

            p_attack = attack_feature / len(attack_records) if len(attack_records) > 0 else 0
            p_normal = normal_feature / len(normal_records) if len(normal_records) > 0 else 0

            mb = max(0, (p_attack - p_normal)) / (p_attack + p_normal + 1e-9)
            md = max(0, (p_normal - p_attack)) / (p_attack + p_normal + 1e-9)

            return min(mb, 1.0), min(md, 1.0)

        kolom_kategorikal = ['conn_state', 'proto', 'service', 'http_status_code',
                           'weird_notice', 'http_method', 'ssl_resumed', 'http_uri']

        for kolom in kolom_kategorikal:
            if kolom not in df.columns:
                continue

            unique_values = df[kolom].value_counts()
            for value, count in unique_values.items():
                support = count / total_records
                if support < min_support:
                    continue

                mb, md = hitung_mb_md(kolom, value)
                confidence = mb / (mb + md + 1e-9)

                if confidence >= min_confidence and (mb > 0.8 or md > 0.8):
                    basis_pengetahuan.append({
                        'kolom': kolom,
                        'kondisi': f"{kolom} == '{value}'",
                        'mb': mb,
                        'md': md
                    })

        kolom_numerik = ['duration', 'src_bytes', 'dst_bytes', 'dns_qtype',
                        'bytes_ratio', 'packet_rate']

        for kolom in kolom_numerik:
            if kolom not in df.columns:
                continue

            q_low = df[kolom].quantile(0.05)
            q_high = df[kolom].quantile(0.95)

            mb_low, md_low = hitung_mb_md(kolom, q_low)
            basis_pengetahuan.append({
                'kolom': kolom,
                'kondisi': f"{kolom} < {q_low}",
                'mb': md_low,
                'md': mb_low
            })

            mb_high, md_high = hitung_mb_md(kolom, q_high)
            basis_pengetahuan.append({
                'kolom': kolom,
                'kondisi': f"{kolom} > {q_high}",
                'mb': mb_high,
                'md': md_high
            })

        self.df_gejala = pd.DataFrame(basis_pengetahuan)
        print(f"✅ Basis pengetahuan dibangun: {len(self.df_gejala)} aturan")

        self.df_gejala.to_csv('basis_pengetahuan_otomatis.csv', index=False)
        print("💾 Disimpan sebagai 'basis_pengetahuan_otomatis.csv'")

    def hitung_cf(self, row_dict):
        """Menghitung Certainty Factor untuk satu baris tanpa eval()."""
        if self.df_gejala is None or len(self.df_gejala) == 0:
            return 0.0

        total_cf = 0.0
        gejala_aktif = 0

        for _, gejala in self.df_gejala.iterrows():
            try:
                kolom = gejala['kolom']
                kondisi_str = gejala['kondisi']
                mb = gejala['mb']
                md = gejala['md']

                if kolom not in row_dict or pd.isna(row_dict[kolom]):
                    continue

                nilai = row_dict[kolom]
                kondisi_terpenuhi = self._evaluasi_kondisi_safe(kondisi_str, kolom, nilai)

                if kondisi_terpenuhi:
                    cf = mb - md
                    if abs(total_cf) < 0.001:
                        total_cf = cf
                    else:
                        total_cf += cf * (1 - abs(total_cf))
                    gejala_aktif += 1
            except Exception:
                continue

        if gejala_aktif > 0:
            total_cf = total_cf / min(gejala_aktif, 10)

        return 1 / (1 + np.exp(-5 * total_cf))

    def _evaluasi_kondisi_safe(self, kondisi_str, kolom, nilai):
        """Evaluasi kondisi string tanpa eval() — hanya dukung ==, <, >."""
        try:
            if ' == ' in kondisi_str:
                parts = kondisi_str.split(' == ')
                value_part = parts[1].strip().strip("'\"")
                return str(nilai).strip().lower() == value_part.lower()

            elif ' < ' in kondisi_str:
                parts = kondisi_str.split(' < ')
                threshold = float(parts[1].strip())
                return pd.to_numeric(nilai, errors='coerce') < threshold

            elif ' > ' in kondisi_str:
                parts = kondisi_str.split(' > ')
                threshold = float(parts[1].strip())
                return pd.to_numeric(nilai, errors='coerce') > threshold
        except (ValueError, TypeError):
            pass

        return False

    def hitung_cf_paralel(self, df):
        """Menghitung CF secara paralel"""
        print("\n⚡ Menghitung skor CF dengan multiprocessing...")

        try:
            num_cores = max(1, cpu_count() - 1)
            chunk_size = max(1, len(df) // (num_cores * 10))
            chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]

            print(f"  - {num_cores} core CPU, {len(chunks)} chunk")

            # Convert df_gejala to plain dicts for pickling
            df_gejala_dicts = self.df_gejala.to_dict('records') if self.df_gejala is not None else []

            # Prepare args: (chunk_dicts, df_gejala_dicts)
            chunk_dicts_list = [chunk.to_dict('records') for chunk in chunks]

            skor_cf = []
            with Pool(num_cores) as pool:
                with tqdm(total=len(chunks), desc="CF Score") as pbar:
                    for result in pool.imap_unordered(
                        _proses_chunk,
                        [(cd, df_gejala_dicts) for cd in chunk_dicts_list]
                    ):
                        skor_cf.extend(result)
                        pbar.update()

            return np.array(skor_cf)
        except Exception as e:
            print(f"⚠️ Parallel processing gagal: {str(e)}")
            print("🔄 Fallback ke mode sequential...")
            return self._hitung_cf_sequential(df)

    def _hitung_cf_sequential(self, df):
        """Fallback sequential CF computation"""
        print("  - Menghitung CF secara sequential...")
        skor = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CF Sequential"):
            skor.append(self.hitung_cf(row.to_dict()))
        return np.array(skor)

    def tentukan_threshold(self, skor_cf):
        """Menentukan threshold adaptif dengan KMeans"""
        print("\n📊 Menentukan threshold adaptif...")

        try:
            skor_array = np.array(skor_cf).reshape(-1, 1)
            unique_values = np.unique(skor_array)

            if len(unique_values) < 3:
                print(f"⚠️ Data hanya memiliki {len(unique_values)} nilai unik — fallback ke threshold default")
                self.thresholds = {'rendah': 0.3, 'sedang': 0.7}
                return False

            n_clusters = min(3, len(unique_values))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(skor_array)
            centers = sorted(kmeans.cluster_centers_.flatten())

            if len(centers) >= 3:
                self.thresholds['rendah'] = centers[0] + (centers[1] - centers[0]) * 0.4
                self.thresholds['sedang'] = centers[1] + (centers[2] - centers[1]) * 0.6
            elif len(centers) == 2:
                self.thresholds['rendah'] = centers[0] + (centers[1] - centers[0]) * 0.3
                self.thresholds['sedang'] = centers[0] + (centers[1] - centers[0]) * 0.7
            else:
                self.thresholds = {'rendah': 0.3, 'sedang': 0.7}

            print(f"✅ Threshold: Rendah={self.thresholds['rendah']:.3f}, Sedang={self.thresholds['sedang']:.3f}")
            return True
        except Exception as e:
            print(f"⚠️ Gagal menentukan threshold adaptif: {str(e)}")
            self.thresholds = {'rendah': 0.3, 'sedang': 0.7}
            return False

    def klasifikasi_risiko(self, skor):
        buffer = 0.05
        if skor < (self.thresholds['rendah'] - buffer):
            return 'rendah'
        elif (self.thresholds['rendah'] + buffer) <= skor < (self.thresholds['sedang'] - buffer):
            return 'sedang'
        elif skor >= (self.thresholds['sedang'] + buffer):
            return 'tinggi'
        else:
            return 'rendah' if skor < self.thresholds['sedang'] else 'sedang'

    def train_xgboost(self, df):
        print("\n🌳 Melatih model XGBoost...")

        try:
            if 'label_binary' not in df.columns:
                print("❌ Kolom 'label_binary' tidak ditemukan")
                return False

            drop_cols = ['label', 'label_binary', 'cf_score', 'tingkat_risiko'] + self.fitur_dihapus
            available_cols = [col for col in drop_cols if col in df.columns]

            X = df.drop(available_cols, axis=1, errors='ignore')
            y = df['label_binary']

            # Encode categorical columns
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

            self.xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                tree_method='hist',
                gamma=0.1,
                reg_alpha=0.5,
                reg_lambda=1
            )

            scores = cross_val_score(
                self.xgb_model, X, y,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            print(f"✅ Cross-val XGBoost: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

            self.xgb_model.fit(X, y)
            return True
        except Exception as e:
            print(f"❌ Error training XGBoost: {str(e)}")
            traceback.print_exc()
            return False

    def train_hybrid_model(self, df):
        print("\n🤖 Melatih model hybrid CF + XGBoost...")

        try:
            if 'cf_score' not in df.columns:
                df['cf_score'] = self.hitung_cf_paralel(df)

            if self.xgb_model is None:
                if not self.train_xgboost(df):
                    raise Exception("XGBoost training failed")

            drop_cols = ['label', 'label_binary', 'cf_score', 'tingkat_risiko'] + self.fitur_dihapus
            available_cols = [col for col in drop_cols if col in df.columns]

            X = df.drop(available_cols, axis=1, errors='ignore')
            y = df['label_binary']

            for col, le in self.label_encoders.items():
                if col in X.columns:
                    X[col] = le.transform(X[col].astype(str))

            xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
            X_hybrid = np.column_stack([xgb_pred, df['cf_score']])

            X_train, X_val, y_train, y_val = train_test_split(
                X_hybrid, y, test_size=0.2, random_state=42, stratify=y)

            self.hybrid_model = XGBClassifier(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.05,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                subsample=0.7,
                colsample_bytree=0.7
            )

            self.hybrid_model.fit(X_train, y_train)

            y_pred = self.hybrid_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            mcc = matthews_corrcoef(y_val, y_pred)

            print(f"✅ Hybrid: acc={accuracy:.4f}, balanced_acc={balanced_acc:.4f}, mcc={mcc:.4f}")
            return True
        except Exception as e:
            print(f"❌ Error training hybrid: {str(e)}")
            traceback.print_exc()
            self.hybrid_model = None
            return False

    def analisis_feature_importance(self, X):
        if self.xgb_model is None:
            return False

        importance = self.xgb_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        high_impact_features = feature_importance[feature_importance['importance'] > 0.3]
        if not high_impact_features.empty:
            print("⚠️ Fitur dominan terdeteksi (mungkin overfitting):")
            print(high_impact_features)
            for fitur in high_impact_features['feature']:
                if fitur not in self.fitur_dihapus:
                    print(f"  - Menghapus fitur: {fitur}")
                    self.fitur_dihapus.append(fitur)
            return True
        return False

    # --- Plotting methods ---

    def plot_cf_distribution(self, df):
        plt.figure(figsize=(10, 6))
        for level, color in [('rendah', 'green'), ('sedang', 'orange'), ('tinggi', 'red')]:
            subset = df[df['tingkat_risiko'] == level]
            if len(subset) > 0:
                sns.kdeplot(subset['cf_score'], label=level.capitalize(),
                           color=color, fill=True)
        plt.axvline(self.thresholds['rendah'], color='blue', linestyle='--', label='Batas Rendah')
        plt.axvline(self.thresholds['sedang'], color='purple', linestyle='--', label='Batas Sedang')
        plt.title('Distribusi Skor CF per Tingkat Risiko')
        plt.xlabel('Skor CF')
        plt.legend()
        plt.savefig('cf_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ cf_distribution.png")

    def plot_risk_distribution(self, df):
        plt.figure(figsize=(8, 6))
        risk_dist = df['tingkat_risiko'].value_counts()
        colors = {'rendah': 'green', 'sedang': 'orange', 'tinggi': 'red'}
        plt.pie(risk_dist, labels=risk_dist.index,
                colors=[colors[x] for x in risk_dist.index],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        plt.title('Persentase Tingkat Risiko')
        plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ risk_distribution.png")

    def plot_confusion_matrix(self, y_true, y_pred, model_name, filename):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Serangan'],
                   yticklabels=['Normal', 'Serangan'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {filename}")

    def plot_roc_curve(self, df):
        plt.figure(figsize=(10, 8))
        if 'cf_score' in df.columns and 'label_binary' in df.columns:
            fpr_cf, tpr_cf, _ = roc_curve(df['label_binary'], df['cf_score'])
            roc_auc_cf = auc(fpr_cf, tpr_cf)
            plt.plot(fpr_cf, tpr_cf, label=f'CF (AUC = {roc_auc_cf:.2f})', linewidth=2)
        if 'prob_xgb' in df.columns and 'label_binary' in df.columns:
            fpr_xgb, tpr_xgb, _ = roc_curve(df['label_binary'], df['prob_xgb'])
            roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
            plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})', linewidth=2)
        if 'prob_hybrid' in df.columns and 'label_binary' in df.columns:
            fpr_hybrid, tpr_hybrid, _ = roc_curve(df['label_binary'], df['prob_hybrid'])
            roc_auc_hybrid = auc(fpr_hybrid, tpr_hybrid)
            plt.plot(fpr_hybrid, tpr_hybrid, label=f'Hybrid (AUC = {roc_auc_hybrid:.2f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Perbandingan Kurva ROC')
        plt.legend(loc="lower right")
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ roc_curves.png")

    def plot_feature_importance(self):
        if self.xgb_model is None:
            return
        plt.figure(figsize=(12, 8))
        plot_importance(self.xgb_model, importance_type='weight')
        plt.title('Feature Importance XGBoost')
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ feature_importance.png")

    def plot_score_comparison(self, df):
        if 'prob_xgb' not in df.columns or 'cf_score' not in df.columns or 'label_binary' not in df.columns:
            return
        plt.figure(figsize=(12, 8))
        sample = df.sample(1000) if len(df) > 1000 else df
        sns.scatterplot(x='prob_xgb', y='cf_score', hue='label_binary',
                       data=sample, alpha=0.6)
        plt.title('Perbandingan Skor XGBoost vs CF')
        plt.xlabel('Probabilitas XGBoost')
        plt.ylabel('Skor CF')
        plt.savefig('score_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ score_comparison.png")

    def evaluasi_model(self, df):
        print("\n📈 Evaluasi hasil...")

        try:
            # Evaluasi CF
            if 'cf_score' in df.columns and 'label_binary' in df.columns:
                print("\n=== Evaluasi Certainty Factor ===")
                precision, recall, thresholds = precision_recall_curve(df['label_binary'], df['cf_score'])
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]

                df['predicted_cf'] = (df['cf_score'] >= optimal_threshold).astype(int)

                cm_cf = confusion_matrix(df['label_binary'], df['predicted_cf'])
                akurasi_cf = accuracy_score(df['label_binary'], df['predicted_cf'])
                balanced_acc_cf = balanced_accuracy_score(df['label_binary'], df['predicted_cf'])
                mcc_cf = matthews_corrcoef(df['label_binary'], df['predicted_cf'])

                print(f"Threshold optimal CF: {optimal_threshold:.4f}")
                print(f"Akurasi CF: {akurasi_cf * 100:.2f}%")
                print(f"Balanced Accuracy CF: {balanced_acc_cf * 100:.2f}%")
                print(f"MCC CF: {mcc_cf:.4f}")
                print("Confusion Matrix CF:")
                print(cm_cf)
                print(classification_report(df['label_binary'], df['predicted_cf'],
                                          target_names=['normal', 'serangan'], zero_division=0))

                self.plot_cf_distribution(df)
                self.plot_risk_distribution(df)
                self.plot_confusion_matrix(df['label_binary'], df['predicted_cf'],
                                         'Certainty Factor', 'cf_confusion_matrix.png')

            # Evaluasi XGBoost
            if self.xgb_model is not None and 'label_binary' in df.columns:
                print("\n=== Evaluasi XGBoost ===")

                drop_cols = ['label', 'label_binary', 'cf_score', 'predicted_cf', 'tingkat_risiko'] + self.fitur_dihapus
                available_cols = [col for col in drop_cols if col in df.columns]

                X = df.drop(available_cols, axis=1, errors='ignore')

                for col, le in self.label_encoders.items():
                    if col in X.columns:
                        try:
                            X[col] = le.transform(X[col].astype(str))
                        except ValueError:
                            X[col] = X[col].apply(lambda x: x if x in le.classes_ else 'unknown')
                            X[col] = le.transform(X[col])

                df['predicted_xgb'] = self.xgb_model.predict(X)
                df['prob_xgb'] = self.xgb_model.predict_proba(X)[:, 1]

                cm_xgb = confusion_matrix(df['label_binary'], df['predicted_xgb'])
                akurasi_xgb = accuracy_score(df['label_binary'], df['predicted_xgb'])
                balanced_acc_xgb = balanced_accuracy_score(df['label_binary'], df['predicted_xgb'])
                mcc_xgb = matthews_corrcoef(df['label_binary'], df['predicted_xgb'])

                print(f"Akurasi XGBoost: {akurasi_xgb * 100:.2f}%")
                print(f"Balanced Accuracy XGBoost: {balanced_acc_xgb * 100:.2f}%")
                print(f"MCC XGBoost: {mcc_xgb:.4f}")
                print("Confusion Matrix XGBoost:")
                print(cm_xgb)
                print(classification_report(df['label_binary'], df['predicted_xgb'],
                                          target_names=['normal', 'serangan'], zero_division=0))

                if self.analisis_feature_importance(X):
                    print("⚠️ Overfitting terdeteksi, melatih ulang tanpa fitur dominan...")
                    self.train_xgboost(df)

                self.plot_confusion_matrix(df['label_binary'], df['predicted_xgb'],
                                         'XGBoost', 'xgb_confusion_matrix.png')
                self.plot_feature_importance()

            # Evaluasi Hybrid
            if (self.hybrid_model is not None and
                'prob_xgb' in df.columns and
                'cf_score' in df.columns and
                'label_binary' in df.columns):
                print("\n=== Evaluasi Hybrid Model ===")
                hybrid_features = np.column_stack([df['prob_xgb'], df['cf_score']])

                df['predicted_hybrid'] = self.hybrid_model.predict(hybrid_features)
                df['prob_hybrid'] = self.hybrid_model.predict_proba(hybrid_features)[:, 1]

                cm_hybrid = confusion_matrix(df['label_binary'], df['predicted_hybrid'])
                akurasi_hybrid = accuracy_score(df['label_binary'], df['predicted_hybrid'])
                balanced_acc_hybrid = balanced_accuracy_score(df['label_binary'], df['predicted_hybrid'])
                mcc_hybrid = matthews_corrcoef(df['label_binary'], df['predicted_hybrid'])

                print(f"Akurasi Hybrid: {akurasi_hybrid * 100:.2f}%")
                print(f"Balanced Accuracy Hybrid: {balanced_acc_hybrid * 100:.2f}%")
                print(f"MCC Hybrid: {mcc_hybrid:.4f}")
                print("Confusion Matrix Hybrid:")
                print(cm_hybrid)
                print(classification_report(df['label_binary'], df['predicted_hybrid'],
                                          target_names=['normal', 'serangan'], zero_division=0))

                self.plot_confusion_matrix(df['label_binary'], df['predicted_hybrid'],
                                         'Hybrid Model', 'hybrid_confusion_matrix.png')

            self.plot_roc_curve(df)
            self.plot_score_comparison(df)

            return True
        except Exception as e:
            print(f"❌ Error evaluasi: {str(e)}")
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Deteksi Serangan dengan CF, XGBoost, dan Hybrid')
    parser.add_argument('dataset', nargs='?', default='train_test_network.csv',
                       help='Path ke dataset CSV (default: train_test_network.csv)')
    args = parser.parse_args()

    sistem = SistemDeteksiAnomali()

    df = sistem.muat_data(args.dataset)
    if df is None:
        exit(1)

    df = sistem.praproses_data(df)
    if df is None:
        exit(1)

    if 'label' not in df.columns:
        print("❌ Kolom 'label' tidak ditemukan")
        exit(1)

    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df['label_binary'] = df['label'].apply(
        lambda x: 1 if x in ['attack', 'malicious', '1', 'true'] else 0)

    sistem.bangun_basis_pengetahuan(df)

    skor_cf = sistem.hitung_cf_paralel(df)
    if skor_cf is None:
        exit(1)
    df['cf_score'] = skor_cf

    sistem.tentukan_threshold(df['cf_score'])
    df['tingkat_risiko'] = df['cf_score'].apply(sistem.klasifikasi_risiko)

    sistem.train_xgboost(df)
    sistem.train_hybrid_model(df)

    print("\n🔎 Distribusi Tingkat Risiko:")
    if 'tingkat_risiko' in df.columns:
        risk_dist = df['tingkat_risiko'].value_counts()
        for level in ['rendah', 'sedang', 'tinggi']:
            if level in risk_dist.index:
                count = risk_dist[level]
                print(f"  - {level.capitalize()}: {count} ({count/len(df)*100:.1f}%)")

    sistem.evaluasi_model(df)

    output_file = 'hasil_prediksi_enhanced.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Hasil disimpan: {output_file}")
    print("🎉 Selesai!")


if __name__ == '__main__':
    main()
