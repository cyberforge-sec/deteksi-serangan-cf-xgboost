
<p align="center">
  <h1 align="center">🛡️ Network Anomaly Detection — CF + XGBoost + Hybrid</h1>
</p>

<p align="center">
  <strong>Cyber-attack detection using Certainty Factor, XGBoost, and hybrid ensemble methods</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/XGBoost-1.5+-orange" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

---

## 📋 Overview

This project implements the **Hybrid Certainty Factor–XGBoost** approach from our published paper:

> **Aprianto, A. D., Maharrani, R. H., Auliya, I. C. R., & Alifiah, V. R. (2026).** A Hybrid Certainty Factor–XGBoost Approach for Cyberattack Detection Using the TON_IoT Dataset. *Journal of Information Systems and Informatics*, 8(2), 1519. https://doi.org/10.63158/journalisi.v8i2.1519

It combines three detection methods:

1. **Certainty Factor (CF)** — rule‑based reasoning derived automatically from training data
2. **XGBoost** — gradient‑boosted decision trees for probabilistic classification
3. **Hybrid Ensemble (meta‑learning)** — meta‑classifier combining CF scores and XGBoost probabilities

The system uses the [**TON_IoT**](https://research.unsw.edu.au/projects/toniot-datasets) dataset from the University of New South Wales.

### Why hybrid?

| Approach | Accuracy (paper) | Strength | Weakness |
|---|---|---|---|
| CF (rules) | 76.31% | Interpretable, fast, no training | Lower accuracy, manual thresholding |
| XGBoost | 99.61% | High accuracy, handles feature interactions | Black‑box, needs large data |
| Hybrid CF–XGBoost | 99.42% | High accuracy + interpretable rules | Slightly more complex pipeline |

---

## 🚀 Quick Start

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
```

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python akurasi.py --dataset path/to/train_test_network.csv
```

If no `--dataset` is provided, defaults to `./train_test_network.csv`.

### Output

All results go to the `output/` directory:

| File | Description |
|---|---|
| `hasil_prediksi_enhanced.csv` | Test‑set predictions with CF scores & risk levels |
| `basis_pengetahuan_otomatis.csv` | CF rules extracted from training data |
| `cf_distribution.png` | KDE of CF scores by risk level |
| `roc_curves.png` | ROC curves for all three models |
| `*_confusion_matrix.png` | Confusion matrix per model |
| `feature_importance.png` | XGBoost feature importance |
| `score_comparison.png` | CF vs. XGBoost scatter |

---

## 🔬 Architecture

```
                    ┌──────────────┐
 Dataset ──────────▶│ Preprocess   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌────────────┐ ┌──────────┐ ┌──────────┐
     │ CF Engine  │ │ XGBoost  │ │ XGBoost  │
     │ (train)    │ │ (train)  │ │ (train)  │
     └─────┬──────┘ └────┬─────┘ └────┬─────┘
           │             │            │
           ▼             ▼            ▼
        CF Score    XGBoost prob    Meta-model
           │             │            │
           └─────────────┼────────────┘
                         ▼
              ┌──────────────────┐
              │   Evaluation     │
              │ (test set only)  │
              └──────────────────┘
```

### Data Flow

1. **Preprocessing** — drop uninformative columns, log‑transform skewed features, engineer `bytes_ratio` and `packet_rate`.
2. **Train/Test Split** — 80/20 stratified split **before** any modelling (prevents data leakage).
3. **CF Engine** — extract rules from training data; score both train and test.
4. **XGBoost** — train on training features; predict on test.
5. **Hybrid** — stack CF score + XGBoost probability → meta XGBoost.
6. **Evaluation** — all metrics reported on the **held‑out test set**.

---

## 📊 Metrics

The system reports:

- **Accuracy** — overall correctness
- **Balanced Accuracy** — handles class imbalance
- **Matthews Correlation Coefficient (MCC)** — robust binary metric (handles imbalance better than F1)
- **Confusion Matrix** — TN / FP / FN / TP breakdown
- **ROC AUC** — discrimination ability

---

## 🧩 Command‑Line Arguments

```
python akurasi.py --help

usage: akurasi.py [-h] [--dataset DATASET] [--test-size TEST_SIZE] [--output OUTPUT]

Network Anomaly Detection — CF + XGBoost + Hybrid

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Path to training CSV dataset (default: train_test_network.csv)
  --test-size TEST_SIZE, -t TEST_SIZE
                        Fraction of data held out for final evaluation (default: 0.2)
  --output OUTPUT, -o OUTPUT
                        Output directory for plots and results (default: output/)
```

---

## 🔧 Development

### Structure

```
deteksi-serangan-cf-xgboost/
├── akurasi.py              # Main pipeline
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── .gitignore
└── README.md
```

### Running tests

```bash
# Synthetic smoke test
python -c "
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'duration': np.random.rand(200),
    'src_bytes': np.random.randint(0, 10000, 200),
    'dst_bytes': np.random.randint(0, 5000, 200),
    'proto': np.random.choice(['tcp','udp','icmp'], 200),
    'service': np.random.choice(['http','dns','ssh','-'], 200),
    'label': np.random.choice(['attack','normal'], 200)
})
df.to_csv('/tmp/smoke_test.csv', index=False)
print('Smoke test dataset created: 200 rows')
"
python akurasi.py -d /tmp/smoke_test.csv -o /tmp/smoke_output
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

## 📚 Reference

- **Aprianto, A. D., Maharrani, R. H., Auliya, I. C. R., & Alifiah, V. R. (2026).** A Hybrid Certainty Factor–XGBoost Approach for Cyberattack Detection Using the TON_IoT Dataset. *Journal of Information Systems and Informatics*, 8(2), 1519. https://doi.org/10.63158/journalisi.v8i2.1519
