import os
import pandas as pd
import numpy as np

def generate_multiple_splits_per_user(min_train_days=3, max_splits=3):
    """
    Generate multiple chronological train/test splits per user, using the most recent days as test points.

    For each user, the function takes their labeled data and:
    - Ensures they have at least (min_train_days + 1) days of data.
    - Produces up to `max_splits` tail-end test splits.
    - For each split, the last N-th day is used as test, and the rest as training.

    Example:
        If a user has 5 samples and max_splits=3:
        - Split 0: train = days 0,1,2,3 | test = day 4
        - Split 1: train = days 0,1,2,4 | test = day 3
        - Split 2: train = days 0,1,3,4 | test = day 2

    Parameters:
        min_train_days (int): Minimum number of training days required to perform splitting.
        max_splits (int): Maximum number of test splits to create per user.
    """
    print("=== Generating Tail-End Train/Test Splits per User ===")

    # Load and combine feature CSVs from Sessions A, B, C
    all_dfs = []
    for name in ['A', 'B', 'C']:
        path = f'features_session_{name}.csv'
        print(f"📂 Loading {path}")
        df = pd.read_csv(path)

        # Rename 'sleep_score' to 'sleep_quality' if needed
        if 'sleep_score' in df.columns:
            df = df.rename(columns={'sleep_score': 'sleep_quality'})

        df['session'] = name  # Track session source
        all_dfs.append(df)

    # Merge all sessions together
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Drop rows without sleep quality labels
    full_df = full_df.dropna(subset=['sleep_quality'])

    users = full_df['uid'].unique()
    print(f"🔍 Found {len(users)} unique users")

    # Create splits for each user
    for uid in users:
        df_user = full_df[full_df['uid'] == uid].copy()
        df_user = df_user.sort_values('label_date')  # Ensure chronological order

        total_days = len(df_user)
        if total_days < min_train_days + 1:
            print(f"⏩ Skipping UID {uid} — Not enough samples ({total_days})")
            continue

        # Determine number of splits to create
        num_splits = min(max_splits, total_days - min_train_days)
        print(f"🔧 UID {uid}: Creating {num_splits} splits (from tail)")

        for i in range(num_splits):
            # Take N-th most recent row as test, rest as train
            test_idx = - (num_splits - i)
            test_df = df_user.iloc[[test_idx]]
            train_df = df_user.drop(df_user.index[test_idx])

            # Save the splits to CSV
            train_df.to_csv(f"train_user_{uid}_split{i}.csv", index=False)
            test_df.to_csv(f"test_user_{uid}_split{i}.csv", index=False)

    print("🎯 Splits generation complete.")
