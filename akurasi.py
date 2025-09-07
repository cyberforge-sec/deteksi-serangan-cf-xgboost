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
import re
import warnings
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, plot_importance

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup visualisasi
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_palette("husl")

class SistemDeteksiAnomali:
    def __init__(self):
        self.df_gejala = None
        self.thresholds = {'rendah': 0.3, 'sedang': 0.7}
        self.xgb_model = None
        self.hybrid_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fitur_dihapus = []  # Menyimpan fitur yang dihapus karena terlalu dominan
    
    def muat_data(self, file_data):
        """Memuat data dan melakukan preprocessing"""
        print("📥 Memulai proses pemuatan data...")
        
        try:
            df = pd.read_csv(file_data)
            print(f"✅ Dataset utama dimuat ({len(df)} baris)")
            return df
        except Exception as e:
            print(f"❌ Gagal memuat dataset: {str(e)}")
            return None
    
    def praproses_data(self, df):
        """Melakukan preprocessing data"""
        print("\n🧹 Memulai preprocessing data...")
        
        # Drop columns yang tidak digunakan (TANPA menghapus kolom 'label')
        cols_to_drop = ['src_ip', 'dst_ip', 'dns_query', 'ssl_subject', 'ssl_issuer',
                       'http_user_agent', 'weird_name', 'weird_addl', 'type', 
                       'dns_qclass']  # TAMBAHKAN DNS_QCLASS DI SINI
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        # Proses kolom numerik
        kolom_numerik = ['duration', 'src_bytes', 'dst_bytes', 'dns_qtype']
        for col in kolom_numerik:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                if df[col].max() > 1e6:
                    df[col] = np.log1p(df[col])
                    print(f"  - Normalisasi log pada {col}")
                
                df[col] = df[col].fillna(0)
                print(f"  - Kolom numerik '{col}' diproses")
        
        # Proses kolom kategorikal
        kolom_kategorikal = ['conn_state', 'proto', 'service', 'http_status_code', 
                           'weird_notice', 'http_method', 'ssl_resumed', 'http_uri']
        for col in kolom_kategorikal:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].replace('nan', 'unknown')
                print(f"  - Kolom kategorikal '{col}' diproses")
        
        # Tambahkan fitur baru
        if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
            df['bytes_ratio'] = np.log1p(df['src_bytes'] / (df['dst_bytes'] + 1))
        if 'duration' in df.columns and 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
            df['packet_rate'] = df['duration'] / (df['src_bytes'] + df['dst_bytes'] + 1)
        print("  - Fitur baru ditambahkan: bytes_ratio dan packet_rate")
        
        return df
    
    def bangun_basis_pengetahuan(self, df, min_support=0.05, min_confidence=0.6):
        """Membangun basis pengetahuan secara otomatis dari data"""
        print("\n🔍 Membangun basis pengetahuan gejala secara otomatis...")
        basis_pengetahuan = []
        
        # Persiapan label
        if 'label_binary' not in df.columns:
            # Pastikan kolom 'label' ada sebelum membuat label_binary
            if 'label' not in df.columns:
                print("❌ Kolom 'label' tidak ditemukan")
                return
                
            df['label_binary'] = df['label'].apply(
                lambda x: 1 if str(x).lower() in ['attack', 'malicious', '1', 'true'] else 0)
        
        total_records = len(df)
        attack_records = df[df['label_binary'] == 1]
        normal_records = df[df['label_binary'] == 0]
        
        # Fungsi untuk menghitung MB dan MD berdasarkan distribusi
        def hitung_mb_md(feature, value):
            # Hitung distribusi untuk nilai tertentu
            total_feature = len(df[df[feature] == value])
            attack_feature = len(attack_records[attack_records[feature] == value])
            normal_feature = len(normal_records[normal_records[feature] == value])
            
            # Hindari pembagian nol
            if total_feature == 0:
                return 0.0, 0.0
            
            # Hitung probabilitas kondisional
            p_attack = attack_feature / len(attack_records) if len(attack_records) > 0 else 0
            p_normal = normal_feature / len(normal_records) if len(normal_records) > 0 else 0
            p_value = total_feature / total_records
            
            # Hitung MB dan MD menggunakan rasio likelihood
            mb = max(0, (p_attack - p_normal)) / (p_attack + p_normal + 1e-9)
            md = max(0, (p_normal - p_attack)) / (p_attack + p_normal + 1e-9)
            
            # Normalisasi
            mb = min(mb, 1.0)
            md = min(md, 1.0)
            
            return mb, md
        
        # Proses kolom kategorikal
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
                
                # Filter aturan dengan confidence yang cukup
                confidence = mb / (mb + md + 1e-9)
                if confidence >= min_confidence and (mb > 0.8 or md > 0.8):
                    basis_pengetahuan.append({
                        'kolom': kolom,
                        'kondisi': f"{kolom} == '{value}'",
                        'mb': mb,
                        'md': md
                    })
        
        # Proses kolom numerik dengan analisis distribusi
        kolom_numerik = ['duration', 'src_bytes', 'dst_bytes', 'dns_qtype', 
                        'bytes_ratio', 'packet_rate']
        
        for kolom in kolom_numerik:
            if kolom not in df.columns:
                continue
                
            # Gunakan quantiles untuk menentukan threshold
            q_low = df[kolom].quantile(0.05)
            q_high = df[kolom].quantile(0.95)
            
            # Aturan untuk nilai rendah
            mb_low, md_low = hitung_mb_md(kolom, q_low)
            basis_pengetahuan.append({
                'kolom': kolom,
                'kondisi': f"{kolom} < {q_low}",
                'mb': md_low,  # Nilai rendah lebih mungkin normal
                'md': mb_low   # Sehingga MD tinggi untuk serangan
            })
            
            # Aturan untuk nilai tinggi
            mb_high, md_high = hitung_mb_md(kolom, q_high)
            basis_pengetahuan.append({
                'kolom': kolom,
                'kondisi': f"{kolom} > {q_high}",
                'mb': mb_high,  # Nilai tinggi lebih mungkin serangan
                'md': md_high
            })
        
        self.df_gejala = pd.DataFrame(basis_pengetahuan)
        print(f"✅ Basis pengetahuan dibangun: {len(self.df_gejala)} aturan")
        
        # Simpan untuk inspeksi
        self.df_gejala.to_csv('basis_pengetahuan_otomatis.csv', index=False)
        print("💾 Basis pengetahuan disimpan sebagai 'basis_pengetahuan_otomatis.csv'")
    
    def hitung_cf(self, row_dict):
        """Menghitung Certainty Factor untuk satu baris"""
        total_cf = 0.0
        gejala_aktif = 0
        
        if self.df_gejala is None or len(self.df_gejala) == 0:
            return 0.0
            
        for _, gejala in self.df_gejala.iterrows():
            try:
                kolom = gejala['kolom']
                kondisi = gejala['kondisi']
                mb = gejala['mb']
                md = gejala['md']
                
                if kolom not in row_dict or pd.isna(row_dict[kolom]):
                    continue
                
                try:
                    # Buat environment yang aman untuk evaluasi
                    env = {'__builtins__': None, 'np': np, 're': re}
                    env[kolom] = row_dict[kolom]
                    
                    # Gunakan try-except untuk evaluasi kondisi
                    if eval(kondisi, {'__builtins__': None}, env):
                        cf = mb - md
                        if abs(total_cf) < 0.001:
                            total_cf = cf
                        else:
                            total_cf += cf * (1 - abs(total_cf))
                        gejala_aktif += 1
                except Exception as e:
                    # Tangani error evaluasi
                    continue
            except Exception as e:
                # Tangani error lainnya
                continue
        
        # Batasi maksimal 10 aturan yang berpengaruh
        if gejala_aktif > 0:
            total_cf = total_cf / min(gejala_aktif, 10)
        
        # Gunakan fungsi sigmoid untuk normalisasi
        return 1 / (1 + np.exp(-5 * total_cf))
    
    def hitung_cf_paralel(self, df):
        """Menghitung CF secara paralel"""
        print("\n⚡ Menghitung skor CF dengan multiprocessing...")
        
        try:
            num_cores = max(1, cpu_count() - 1)
            chunk_size = max(1, len(df) // (num_cores * 10))
            chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
            
            print(f"  - Menggunakan {num_cores} core CPU")
            print(f"  - Jumlah chunk: {len(chunks)}")
            print(f"  - Ukuran chunk: {chunk_size} baris")
            
            skor_cf = []
            with Pool(num_cores) as pool:
                worker = partial(self._proses_chunk)
                with tqdm(total=len(chunks), desc="Proses") as pbar:
                    for result in pool.imap_unordered(worker, chunks):
                        skor_cf.extend(result)
                        pbar.update()
            
            return np.array(skor_cf)
        except Exception as e:
            print(f"❌ Error dalam parallel processing: {str(e)}")
            traceback.print_exc()
            return None
    
    def _proses_chunk(self, chunk):
        """Helper function untuk multiprocessing"""
        return [self.hitung_cf(row.to_dict()) for _, row in chunk.iterrows()]
    
    def tentukan_threshold(self, skor_cf):
        """Menentukan threshold adaptif"""
        print("\n📊 Menentukan threshold adaptif...")
        
        try:
            skor_array = np.array(skor_cf).reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, n_init=10).fit(skor_array)
            centers = sorted(kmeans.cluster_centers_.flatten())
            
            self.thresholds['rendah'] = centers[0] + (centers[1] - centers[0]) * 0.4
            self.thresholds['sedang'] = centers[1] + (centers[2] - centers[1]) * 0.6
            
            print(f"✅ Threshold adaptif: Rendah={self.thresholds['rendah']:.3f}, Sedang={self.thresholds['sedang']:.3f}")
            return True
        except Exception as e:
            print(f"⚠️ Gunakan threshold default: {str(e)}")
            self.thresholds = {'rendah': 0.3, 'sedang': 0.7}
            return False
    
    def klasifikasi_risiko(self, skor):
        """Klasifikasi tingkat risiko dengan zona buffer"""
        buffer = 0.05
        
        if skor < (self.thresholds['rendah'] - buffer):
            return 'rendah'
        elif (self.thresholds['rendah'] + buffer) <= skor < (self.thresholds['sedang'] - buffer):
            return 'sedang'
        elif skor >= (self.thresholds['sedang'] + buffer):
            return 'tinggi'
        else:
            if skor < self.thresholds['sedang']:
                return 'rendah'
            else:
                return 'sedang'
    
    def train_xgboost(self, df):
        """Melatih model XGBoost dengan validasi silang dan regularisasi"""
        print("\n🌳 Melatih model XGBoost...")
        
        try:
            if 'label_binary' not in df.columns:
                print("❌ Kolom 'label_binary' tidak ditemukan")
                return False
                
            # Pastikan tidak menghapus kolom yang tidak ada
            drop_cols = ['label', 'label_binary', 'cf_score', 'tingkat_risiko'] + self.fitur_dihapus
            available_cols = [col for col in drop_cols if col in df.columns]
            
            X = df.drop(available_cols, axis=1, errors='ignore')
            y = df['label_binary']
            
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            # Konfigurasi model dengan regularisasi
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
            
            # Validasi silang
            scores = cross_val_score(
                self.xgb_model, X, y, 
                cv=5,  # 5-fold cross validation
                scoring='accuracy',
                n_jobs=-1
            )
            print(f"✅ Validasi Silang XGBoost: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            
            # Latih model dengan semua data
            self.xgb_model.fit(X, y)
            
            return True
        except Exception as e:
            print(f"❌ Error training XGBoost: {str(e)}")
            traceback.print_exc()
            return False
    
    def train_hybrid_model(self, df):
        """Melatih model hybrid CF + XGBoost dengan regularisasi"""
        print("\n🤖 Melatih model hybrid CF + XGBoost...")
        
        try:
            if 'cf_score' not in df.columns:
                df['cf_score'] = self.hitung_cf_paralel(df)
            
            if self.xgb_model is None:
                if not self.train_xgboost(df):
                    raise Exception("XGBoost training failed")
            
            # Pastikan tidak menghapus kolom yang tidak ada
            drop_cols = ['label', 'label_binary', 'cf_score', 'tingkat_risiko'] + self.fitur_dihapus
            available_cols = [col for col in drop_cols if col in df.columns]
            
            X = df.drop(available_cols, axis=1, errors='ignore')
            y = df['label_binary']
            
            for col, le in self.label_encoders.items():
                if col in X.columns:
                    X[col] = le.transform(X[col].astype(str))
            
            xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
            X_hybrid = np.column_stack([xgb_pred, df['cf_score']])
            
            # Pisahkan data validasi
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
            
            # PERBAIKAN: Hapus parameter yang tidak didukung
            self.hybrid_model.fit(X_train, y_train)
            
            y_pred = self.hybrid_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            mcc = matthews_corrcoef(y_val, y_pred)
            
            print(f"✅ Hybrid model trained with validation accuracy: {accuracy:.4f}")
            print(f"  - Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  - Matthews Correlation Coefficient: {mcc:.4f}")
            
            return True
        except Exception as e:
            print(f"❌ Error training hybrid model: {str(e)}")
            traceback.print_exc()
            self.hybrid_model = None  # Pastikan di-set None jika gagal
            return False
    
    def analisis_feature_importance(self, X):
        """Analisis feature importance untuk deteksi overfitting dan hapus fitur dominan"""
        if self.xgb_model is None:
            return False
            
        importance = self.xgb_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Identifikasi fitur dengan importance sangat tinggi
        high_impact_features = feature_importance[feature_importance['importance'] > 0.3]
        if not high_impact_features.empty:
            print("⚠️ Fitur dengan dampak sangat tinggi (mungkin penyebab overfitting):")
            print(high_impact_features)
            
            # Tambahkan fitur yang terlalu dominan ke daftar penghapusan
            for fitur in high_impact_features['feature']:
                if fitur not in self.fitur_dihapus:
                    print(f"  - Hapus fitur: {fitur}")
                    self.fitur_dihapus.append(fitur)
            
            return True
        
        return False
    
    def plot_cf_distribution(self, df):
        """Visualisasi distribusi skor CF"""
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
        print("✅ Distribusi CF disimpan sebagai 'cf_distribution.png'")
    
    def plot_risk_distribution(self, df):
        """Visualisasi distribusi risiko"""
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
        print("✅ Distribusi risiko disimpan sebagai 'risk_distribution.png'")
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, filename):
        """Visualisasi confusion matrix untuk model tertentu"""
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
        print(f"✅ Confusion matrix {model_name} disimpan sebagai '{filename}'")
    
    def plot_roc_curve(self, df):
        """Visualisasi kurva ROC untuk semua model"""
        plt.figure(figsize=(10, 8))
        
        # CF ROC
        if 'cf_score' in df.columns and 'label_binary' in df.columns:
            fpr_cf, tpr_cf, _ = roc_curve(df['label_binary'], df['cf_score'])
            roc_auc_cf = auc(fpr_cf, tpr_cf)
            plt.plot(fpr_cf, tpr_cf, label=f'CF (AUC = {roc_auc_cf:.2f})', linewidth=2)
        
        # XGBoost ROC
        if 'prob_xgb' in df.columns and 'label_binary' in df.columns:
            fpr_xgb, tpr_xgb, _ = roc_curve(df['label_binary'], df['prob_xgb'])
            roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
            plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})', linewidth=2)
        
        # Hybrid ROC
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
        print("✅ Kurva ROC disimpan sebagai 'roc_curves.png'")
    
    def plot_feature_importance(self):
        """Visualisasi feature importance XGBoost"""
        if self.xgb_model is None:
            return
            
        plt.figure(figsize=(12, 8))
        plot_importance(self.xgb_model, importance_type='weight')
        plt.title('Feature Importance XGBoost')
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Feature importance disimpan sebagai 'feature_importance.png'")
    
    def plot_score_comparison(self, df):
        """Visualisasi perbandingan skor CF vs XGBoost"""
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
        print("✅ Perbandingan skor disimpan sebagai 'score_comparison.png'")
    
    def evaluasi_model(self, df):
        """Evaluasi performa model dengan visualisasi terpisah"""
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
                cr_cf = classification_report(df['label_binary'], df['predicted_cf'],
                                             target_names=['normal','serangan'],
                                             zero_division=0)
                
                print(f"\nOptimal Threshold CF: {optimal_threshold:.4f}")
                print(f"✅ Akurasi CF: {akurasi_cf * 100:.2f}%")
                print(f"✅ Balanced Accuracy CF: {balanced_acc_cf * 100:.2f}%")
                print(f"✅ Matthews Correlation Coefficient CF: {mcc_cf:.4f}")
                print("Confusion Matrix CF:")
                print(cm_cf)
                print("\nClassification Report CF:")
                print(cr_cf)
                
                # Visualisasi CF
                self.plot_cf_distribution(df)
                self.plot_risk_distribution(df)
                self.plot_confusion_matrix(df['label_binary'], df['predicted_cf'], 
                                         'Certainty Factor', 'cf_confusion_matrix.png')
            
            # Evaluasi XGBoost
            if self.xgb_model is not None and 'label_binary' in df.columns:
                print("\n=== Evaluasi XGBoost ===")
                
                # Pastikan tidak menghapus kolom yang tidak ada
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
                cr_xgb = classification_report(df['label_binary'], df['predicted_xgb'],
                                             target_names=['normal','serangan'],
                                             zero_division=0)
                
                print(f"✅ Akurasi XGBoost: {akurasi_xgb * 100:.2f}%")
                print(f"✅ Balanced Accuracy XGBoost: {balanced_acc_xgb * 100:.2f}%")
                print(f"✅ Matthews Correlation Coefficient XGBoost: {mcc_xgb:.4f}")
                print("Confusion Matrix XGBoost:")
                print(cm_xgb)
                print("\nClassification Report XGBoost:")
                print(cr_xgb)
                
                # Analisis feature importance
                if self.analisis_feature_importance(X):
                    print("⚠️ Deteksi fitur yang berpotensi menyebabkan overfitting")
                    print("🔄 Melatih ulang model tanpa fitur dominan...")
                    if self.train_xgboost(df):  # Latih ulang tanpa fitur dominan
                        print("✅ Model XGBoost dilatih ulang tanpa fitur dominan")
                
                # Visualisasi XGBoost
                self.plot_confusion_matrix(df['label_binary'], df['predicted_xgb'], 
                                         'XGBoost', 'xgb_confusion_matrix.png')
                self.plot_feature_importance()
            
            # Evaluasi Hybrid Model - HANYA jika model hybrid berhasil dilatih
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
                cr_hybrid = classification_report(df['label_binary'], df['predicted_hybrid'],
                                                target_names=['normal','serangan'],
                                                zero_division=0)
                
                print(f"✅ Akurasi Hybrid: {akurasi_hybrid * 100:.2f}%")
                print(f"✅ Balanced Accuracy Hybrid: {balanced_acc_hybrid * 100:.2f}%")
                print(f"✅ Matthews Correlation Coefficient Hybrid: {mcc_hybrid:.4f}")
                print("Confusion Matrix Hybrid:")
                print(cm_hybrid)
                print("\nClassification Report Hybrid:")
                print(cr_hybrid)
                
                # Visualisasi Hybrid
                self.plot_confusion_matrix(df['label_binary'], df['predicted_hybrid'], 
                                         'Hybrid Model', 'hybrid_confusion_matrix.png')
            
            # Visualisasi tambahan
            self.plot_roc_curve(df)
            self.plot_score_comparison(df)
            
            return True
        except Exception as e:
            print(f"❌ Error evaluasi: {str(e)}")
            traceback.print_exc()
            return False

def main():
    # Inisialisasi sistem
    sistem = SistemDeteksiAnomali()
    
    # 1. Muat data
    df = sistem.muat_data('train_test_network.csv')
    if df is None:
        exit(1)
    
    # 2. Preprocessing data (TANPA menghapus kolom 'label')
    df = sistem.praproses_data(df)
    if df is None:
        exit(1)
    
    # 3. Persiapan label (pastikan kolom 'label' ada)
    if 'label' not in df.columns:
        print("❌ Kolom 'label' tidak ditemukan setelah preprocessing")
        exit(1)
        
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df['label_binary'] = df['label'].apply(
        lambda x: 1 if x in ['attack', 'malicious', '1', 'true'] else 0)
    
    # 4. Bangun basis pengetahuan secara otomatis
    sistem.bangun_basis_pengetahuan(df)
    
    # 5. Hitung skor CF
    skor_cf = sistem.hitung_cf_paralel(df)
    if skor_cf is None:
        exit(1)
    df['cf_score'] = skor_cf
    
    # 6. Tentukan threshold adaptif
    sistem.tentukan_threshold(df['cf_score'])
    
    # 7. Klasifikasi risiko
    df['tingkat_risiko'] = df['cf_score'].apply(sistem.klasifikasi_risiko)
    
    # 8. Latih model XGBoost
    sistem.train_xgboost(df)
    
    # 9. Latih model hybrid
    sistem.train_hybrid_model(df)
    
    # 10. Analisis distribusi risiko
    print("\n🔎 Distribusi Tingkat Risiko:")
    if 'tingkat_risiko' in df.columns:
        risk_dist = df['tingkat_risiko'].value_counts()
        for level in ['rendah', 'sedang', 'tinggi']:
            if level in risk_dist.index:
                count = risk_dist[level]
                print(f"  - {level.capitalize()}: {count} ({count/len(df)*100:.1f}%)")
    
    # 11. Evaluasi model
    sistem.evaluasi_model(df)
    
    # 12. Simpan hasil
    output_file = 'hasil_prediksi_enhanced.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Hasil disimpan ke: {output_file}")
    print("🎉 Proses selesai dengan sukses!")

if __name__ == '__main__':
    main()