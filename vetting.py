import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from itertools import combinations

def vet_features_spearman_per_user():
    """
    Perform per-user feature vetting based on Spearman correlation.

    Steps:
    1. Load all per-user train/test CSV files (with split index) that haven't been vetted yet.
    2. Remove highly correlated feature pairs (|rho| > 0.8), keeping the one with stronger correlation to the label.
    3. Rank remaining features by absolute Spearman correlation with sleep quality.
    4. Keep the top 15 features.
    5. Normalize features using StandardScaler (fit on train, apply to test).
    6. Save vetted and normalized train/test CSV files.

    Output files:
        - train_user_<uid>_split<i>_vetted.csv
        - test_user_<uid>_split<i>_vetted.csv
    """
    print("=== Running Per-User Feature Vetting ===")

    # Iterate over all relevant train files
    for filename in os.listdir():
        if filename.startswith("train_user_") and "split" in filename and filename.endswith(".csv") and "_vetted" not in filename:
            split_part = filename.split("split")[1].split(".")[0]
            uid = filename.split("_")[2]
            train_path = filename
            test_path = f"test_user_{uid}_split{split_part}.csv"

            if not os.path.exists(test_path):
                print(f"❌ Test file for UID {uid} split {split_part} not found.")
                continue

            print(f"🔍 Vetting UID {uid} — split {split_part}")

            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            drop_cols = ['sleep_quality', 'uid', 'label_date']
            feature_cols = [col for col in df_train.select_dtypes(include=[np.number]).columns if col not in drop_cols]

            if len(feature_cols) < 2:
                print(f"⚠️ UID {uid} split {split_part} — Not enough numeric features, skipping.")
                continue

            # ===== STEP 1: Remove highly correlated feature pairs =====
            df = df_train.copy()
            to_remove = set()
            target_col = 'sleep_quality'

            for f1, f2 in combinations(feature_cols, 2):
                if f1 in to_remove or f2 in to_remove:
                    continue
                try:
                    corr = df[[f1, f2]].corr(method='spearman').iloc[0, 1]
                    if abs(corr) > 0.8:
                        # Retain feature with stronger correlation to label
                        corr_f1 = abs(spearmanr(df[f1], df[target_col])[0])
                        corr_f2 = abs(spearmanr(df[f2], df[target_col])[0])
                        if corr_f1 < corr_f2:
                            to_remove.add(f1)
                        else:
                            to_remove.add(f2)
                except Exception as e:
                    print(f"⚠️ Skipping pair ({f1}, {f2}) due to error: {e}")
                    continue

            kept_features = [f for f in feature_cols if f not in to_remove]
            print(f"🧹 Removed {len(to_remove)} highly correlated features.")

            # ===== STEP 2: Rank features by absolute Spearman correlation to label =====
            correlations = []
            for col in kept_features:
                x = df[col]
                y = df[target_col]
                if x.nunique() <= 1 or x.std() < 1e-6:
                    continue
                corr, _ = spearmanr(x, y)
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))

            # Keep top 15 features
            top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:15]
            selected_features = [f[0] for f in top_features]

            if len(selected_features) == 0:
                print(f"⚠️ UID {uid} split {split_part} — no valid features found, skipping.")
                continue

            # ===== STEP 3: Normalize features =====
            scaler = StandardScaler()
            try:
                X_train_scaled = scaler.fit_transform(df_train[selected_features])
                X_test_scaled = scaler.transform(df_test[selected_features])
            except ValueError as e:
                print(f"⚠️ UID {uid} split {split_part} — Skipping due to normalization error: {e}")
                continue

            # ===== STEP 4: Save vetted output =====
            df_train_out = pd.DataFrame(X_train_scaled, columns=selected_features)
            df_train_out['sleep_quality'] = df_train['sleep_quality'].values
            df_train_out['uid'] = df_train['uid'].values
            df_train_out['label_date'] = df_train['label_date'].values
            df_train_out.to_csv(f"train_user_{uid}_split{split_part}_vetted.csv", index=False)

            df_test_out = pd.DataFrame(X_test_scaled, columns=selected_features)
            df_test_out['sleep_quality'] = df_test['sleep_quality'].values
            df_test_out['uid'] = df_test['uid'].values
            df_test_out['label_date'] = df_test['label_date'].values
            df_test_out.to_csv(f"test_user_{uid}_split{split_part}_vetted.csv", index=False)

            print(f"✅ UID {uid} split {split_part} — vetted data saved.")

    print("\n🎉 Per-user feature vetting complete.")
