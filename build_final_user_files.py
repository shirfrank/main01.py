import os
import pandas as pd

def build_final_user_files(unified_folder='final_user_files', data_folder='.', output_folder='final_user_csvs'):
    """
    Builds final unified training and testing CSV files per user.

    Each user has multiple train/test splits. For each user, this function:
    - Loads the list of final selected features from `unified_folder`
    - Loads all train/test split files for that user from `data_folder`
    - Retains only the selected features + sleep_quality in each split
    - Concatenates all train splits into a final training file
    - Concatenates all test splits into a final testing file
    - Saves the result to `output_folder`

    Parameters:
    ----------
    unified_folder : str
        Directory containing per-user unified selected feature lists (from feature selection stage).
        Files should be named like `unified_features_uid_XXX.csv`

    data_folder : str
        Directory containing the vetted and selected split files.
        Files should be named like `selected_train_user_UID_splitX.csv`, `selected_test_user_UID_splitX.csv`.

    output_folder : str
        Directory where the final per-user train/test files will be saved.
    """

    os.makedirs(output_folder, exist_ok=True)
    print("📦 Building final unified train/test files per user...")

    # Loop over each user feature list in the unified folder
    for fname in os.listdir(unified_folder):
        if not fname.startswith('unified_features_uid_') or not fname.endswith('.csv'):
            continue

        uid = fname.split('_')[-1].replace('.csv', '')  # Extract user ID
        unified_path = os.path.join(unified_folder, fname)

        try:
            # Load the selected features list for the current user
            selected_features = pd.read_csv(unified_path, header=None)[0].tolist()
        except Exception as e:
            print(f"❌ Failed to read unified feature list for UID {uid}: {e}")
            continue

        # Find all vetted train/test split files for this user
        train_files = [
            f for f in os.listdir(data_folder)
            if f.startswith(f"selected_train_user_{uid}_split") and f.endswith(".csv")
        ]
        test_files = [
            f for f in os.listdir(data_folder)
            if f.startswith(f"selected_test_user_{uid}_split") and f.endswith(".csv")
        ]

        if not train_files or not test_files:
            print(f"⚠️  UID {uid} — Missing selected split files. Skipping.")
            continue

        train_dfs = []
        test_dfs = []

        # Process all train splits
        for tfile in sorted(train_files):
            tpath = os.path.join(data_folder, tfile)
            df = pd.read_csv(tpath)
            # Retain only selected features + label
            columns = [c for c in selected_features if c in df.columns] + ['sleep_quality']
            train_dfs.append(df[columns])

        # Process all test splits
        for tfile in sorted(test_files):
            tpath = os.path.join(data_folder, tfile)
            df = pd.read_csv(tpath)
            # Retain only selected features + label
            columns = [c for c in selected_features if c in df.columns] + ['sleep_quality']
            test_dfs.append(df[columns])

        try:
            # Concatenate all train and test splits
            train_final = pd.concat(train_dfs, ignore_index=True)
            test_final = pd.concat(test_dfs, ignore_index=True)

            # Save final per-user train and test files
            train_final.to_csv(os.path.join(output_folder, f"train_user_{uid}.csv"), index=False)
            test_final.to_csv(os.path.join(output_folder, f"test_user_{uid}.csv"), index=False)

            print(f"✅ UID {uid}: Final train/test files created.")

        except Exception as e:
            print(f"❌ Failed to merge final files for UID {uid}: {e}")

    print("\n🎯 Final user files ready for modeling.")
