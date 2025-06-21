import os
import pandas as pd
from collections import Counter

def merge_selected_features(selected_folder, output_folder, top_n=5):
    """
    Merges the most frequently selected features across multiple splits per user.

    Parameters:
    - selected_folder (str): Path to the folder containing selected feature CSVs (e.g., selected_train_user_...).
    - output_folder (str): Path to the folder where unified feature lists will be saved.
    - top_n (int): Number of top most common features to keep per user.

    Output:
    - For each user, saves a CSV file named 'unified_features_uid_<UID>.csv'
      containing the top_n most frequent features across splits.
    """
    print("📊 Merging selected features per user...")

    selected_files = [f for f in os.listdir(selected_folder) if f.startswith('selected_train_user_') and 'split' in f]

    user_splits = {}
    for f in selected_files:
        parts = f.split('_')
        uid = parts[3]
        user_splits.setdefault(uid, []).append(f)

    for uid, files in user_splits.items():
        print(f"\n🔍 UID {uid} — found {len(files)} splits")

        feature_counter = Counter()

        for file in files:
            path = os.path.join(selected_folder, file)
            df = pd.read_csv(path)
            features = [col for col in df.columns if col not in ['sleep_quality', 'uid', 'label_date']]
            feature_counter.update(features)

        final_features = [f for f, _ in feature_counter.most_common(top_n)]
        print(f"✅ UID {uid} — Final top {top_n} features: {final_features}")

        output_path = os.path.join(output_folder, f'unified_features_uid_{uid}.csv')
        pd.Series(final_features).to_csv(output_path, index=False)

    print("\n🎯 Feature merging complete.")
