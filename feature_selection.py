import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from itertools import combinations
from tqdm import tqdm

def exhaustive_feature_selection(input_folder, output_folder, max_features=5):
    """
    Performs exhaustive wrapper-based feature selection using Ridge regression and TimeSeriesSplit cross-validation.

    For each user's vetted train/test split (e.g., 'train_user_123_split1_vetted.csv'), the function tries all
    combinations of up to `max_features` features and evaluates them using the mean squared error (MSE).
    The best-performing feature combination is selected.

    Selected features are saved into new CSV files for both train and test sets in the output directory.

    Parameters:
        input_folder (str): Folder containing vetted training/testing files for each user split.
        output_folder (str): Folder to save the train/test CSVs with selected features.
        max_features (int): Maximum number of features to try in each combination.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Collect all vetted training files with a split tag
    train_files = sorted([
        f for f in os.listdir(input_folder)
        if f.startswith('train_user_') and 'split' in f and f.endswith('_vetted.csv')
    ])

    # Loop through each train file per user and split
    for train_file in train_files:
        parts = train_file.split('_')
        uid = parts[2]  # Extract user ID
        split_part = train_file.split('split')[1].split('_')[0]  # Extract split number
        test_file = f'test_user_{uid}_split{split_part}_vetted.csv'
        test_file_path = os.path.join(input_folder, test_file)

        # Skip if corresponding test file is missing
        if not os.path.exists(test_file_path):
            print(f"❌ Skipping UID {uid} split {split_part} (no matching test file)")
            continue

        print(f"\n🔍 Selecting features for UID {uid} split {split_part}...")

        # Load training and test data
        df_train = pd.read_csv(os.path.join(input_folder, train_file)).dropna(subset=['sleep_quality'])
        df_test = pd.read_csv(test_file_path).dropna(subset=['sleep_quality'])

        # Sort train set by label_date to respect temporal order for TimeSeriesSplit
        df_train = df_train.sort_values(by="label_date")

        # Select numeric columns only, excluding target and ID fields
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['uid', 'sleep_quality']]

        X_train_df = df_train[feature_cols]
        X_test_df = df_test[feature_cols]
        y_train = df_train['sleep_quality'].values
        y_test = df_test['sleep_quality'].values

        # Number of splits for TimeSeriesSplit
        n_samples = len(X_train_df)
        n_splits = min(3, n_samples - 1)

        # Skip splits with insufficient data
        if n_splits < 2:
            print(f"⚠️ Skipping UID {uid} split {split_part} — not enough samples (samples={n_samples})")
            continue

        tscv = TimeSeriesSplit(n_splits=n_splits)

        best_score = np.inf
        best_features = []

        # Try all feature combinations of size 1 to max_features
        for k in range(1, max_features + 1):
            for combo in tqdm(combinations(feature_cols, k), desc=f"UID {uid} | split {split_part} | k={k}"):
                model = Ridge()
                try:
                    # Use negative MSE for scoring (so lower is better)
                    scores = cross_val_score(model, X_train_df[list(combo)], y_train,
                                             cv=tscv, scoring='neg_mean_squared_error')
                    mse = -np.mean(scores)
                    # Keep track of the best performing feature set
                    if mse < best_score:
                        best_score = mse
                        best_features = list(combo)
                except ValueError as e:
                    print(f"⚠️ UID {uid} | combo {combo} failed: {e}")
                    continue

        if not best_features:
            print(f"⚠️ UID {uid} split {split_part}: No valid feature combinations found.")
            continue

        print(f"✅ UID {uid} split {split_part}: Best features: {best_features} | MSE = {best_score:.3f}")

        # Save selected features + target/metadata for train/test sets
        train_selected = df_train[best_features + ['sleep_quality', 'uid', 'label_date']]
        test_selected = df_test[best_features + ['sleep_quality', 'uid', 'label_date']]

        train_out = os.path.join(output_folder, f'selected_train_user_{uid}_split{split_part}.csv')
        test_out = os.path.join(output_folder, f'selected_test_user_{uid}_split{split_part}.csv')
        train_selected.to_csv(train_out, index=False)
        test_selected.to_csv(test_out, index=False)

    print("\n🎯 Exhaustive feature selection completed for all users and splits.")
