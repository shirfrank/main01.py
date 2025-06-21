import os
import pandas as pd
# === Import modules for each stage ===
from explore_sensors import explore_sensors
from get_sensor_window import estimate_sleep_wake_times
from clean import clean_accel, clean_light, clean_screen, clean_wifi, clean_location, clean_calls
from feature_extraction_v2 import (compute_accel_features, compute_light_features, compute_screen_features,
    compute_wifi_features, compute_location_features, compute_calls_features,
    compute_circadian_features)
from train_test import generate_multiple_splits_per_user
from vetting import vet_features_spearman_per_user
from feature_selection import exhaustive_feature_selection
from merge_features import merge_selected_features
from build_final_user_files import build_final_user_files
# === Stage Execution Flags ===
run_clean_extract = True
run_train = True
run_vetting = True
run_selection = True
run_merge = True
#shirr

# === Paths Setup ===
data_folder = 'data'
session_a_folder = os.path.join(data_folder, 'Session A')
session_b_folder = os.path.join(data_folder, 'Session B')
session_c_folder = os.path.join(data_folder, 'Session C')

sessions = [
    {
        'name': 'A',
        'sensor_path': os.path.join(session_a_folder, 'bhq_hisha_2025.xlsx'),
        'label_path': os.path.join(session_a_folder, 'Session_A_Label.csv')
    },
    {
        'name': 'B',
        'sensor_path': os.path.join(session_b_folder, 'bhq_hisha_2025_s2.xlsx'),
        'label_path': os.path.join(session_b_folder, 'Session_B_Label.csv')
    },
    {
        'name': 'C',
        'sensor_path': os.path.join(session_c_folder, 'bhq_hisha_2025_s3.xlsx'),
        'label_path': os.path.join(session_c_folder, 'Session_C_Label.csv')
    }
]

# === Helper Function to Extract from Tuple ===
def extract_df(result):
    """If result is a tuple (DataFrame, meta), return the DataFrame."""
    return result[0] if isinstance(result, tuple) else result

# === STAGE 1: Clean & Extract Features ===
if run_clean_extract:
    for session in sessions:
        name = session['name']
        print(f"\n=== Processing Session {name} ===")

        # Load sensor and label data
        sensor_df = pd.read_excel(session['sensor_path'])
        label_df = pd.read_csv(session['label_path'])

        explore_sensors(sensor_df, f"Session {name}")

        sensor_df['datetime'] = pd.to_datetime(sensor_df['datetime'])
        sensor_df['date'] = sensor_df['datetime'].dt.date

        print(f"\nChecking missing values in Session {name}:")
        print(sensor_df.isnull().sum())

        # Clean and separate each sensor
        accel_df = extract_df(clean_accel(sensor_df))
        light_df = extract_df(clean_light(sensor_df))
        screen_df = extract_df(clean_screen(sensor_df))
        wifi_df = extract_df(clean_wifi(sensor_df))
        location_df = extract_df(clean_location(sensor_df))
        calls_df = extract_df(clean_calls(sensor_df))

        # Optionally save cleaned data
        accel_df.to_csv(f"cleaned_accel_session_{name}.csv", index=False)
        light_df.to_csv(f"cleaned_light_session_{name}.csv", index=False)
        screen_df.to_csv(f"cleaned_screen_session_{name}.csv", index=False)
        wifi_df.to_csv(f"cleaned_wifi_session_{name}.csv", index=False)
        location_df.to_csv(f"cleaned_location_session_{name}.csv", index=False)
        calls_df.to_csv(f"cleaned_calls_session_{name}.csv", index=False)

        features_list = []
        evaluations = []
        label_df['Timestamp'] = pd.to_datetime(label_df['Timestamp'], dayfirst=True)
        unique_dates = label_df['Timestamp'].dt.date.unique()

        for date in unique_dates:
            day_rows = label_df[label_df['Timestamp'].dt.date == date]
            for _, row in day_rows.iterrows():
                uid = row['uid']
                label_date = row['Timestamp']
                sleep_score = row['Rate your overall sleep last night:']

                # Estimate subwindows (day, pre_sleep, sleep)
                windows, evaluation = estimate_sleep_wake_times(
                    screen_df, light_df, accel_df, location_df, wifi_df, label_df, uid, label_date
                )
                if windows is None:
                    continue
                if evaluation:
                    evaluations.append(evaluation)

                start_time, end_time = windows['full_window']
                sensor_filter = lambda df: df[
                    (df['uid'] == uid) & (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
                ].copy()

                # Compute all features and combine into one row
                row_features = {
                    'uid': uid,
                    'label_date': label_date,
                    'sleep_score': sleep_score
                }
                row_features.update(compute_accel_features(sensor_filter(accel_df)))
                row_features.update(compute_light_features(sensor_filter(light_df), windows))
                row_features.update(compute_screen_features(sensor_filter(screen_df), sensor_filter(light_df), windows))
                row_features.update(compute_wifi_features(sensor_filter(wifi_df), windows))
                row_features.update(compute_location_features(sensor_filter(location_df), windows))
                row_features.update(compute_calls_features(sensor_filter(calls_df), windows))
                row_features.update(compute_circadian_features(
                    sensor_filter(screen_df), sensor_filter(light_df), sensor_filter(accel_df), windows
                ))

                features_list.append(row_features)

        features_df = pd.DataFrame(features_list)
        output_path = f"features_session_{name}.csv"
        features_df.to_csv(output_path, index=False)

        print(f"\n✅ Features saved to {output_path}")
        print(f"Shape: {features_df.shape}")
        print("NaNs:\n", features_df.isnull().sum())

        if evaluations:
            pd.DataFrame(evaluations).to_csv(f"evaluation_report_session_{name}.csv", index=False)
            print(f"📊 Evaluation report saved to evaluation_report_session_{name}.csv")

# === STAGE 2: Train/Test Splitting ===
if run_train:
    print("\n=== Running Train/Test Split ===")
    generate_multiple_splits_per_user()

# === STAGE 3: Feature Vetting (Correlation and Top Ranking) ===
if run_vetting:
    print("\n=== Running Feature Vetting ===")
    vet_features_spearman_per_user()

# === STAGE 4: Wrapper Feature Selection (Greedy Exhaustive Search) ===
if run_selection:
    print("\n=== Checking for existing feature selection files ===")
    selected_dir = 'selected_features'
    os.makedirs(selected_dir, exist_ok=True)

    existing_selected = [
        f for f in os.listdir(selected_dir)
        if f.startswith("selected_train_user_") and f.endswith(".csv")
    ]

    if len(existing_selected) > 0:
        print(f"📁 Found {len(existing_selected)} existing selected feature files — skipping selection step.")
    else:
        print("\n=== Running Wrapper Feature Selection ===")
        exhaustive_feature_selection(
            input_folder='.',
            output_folder=selected_dir,
            max_features=15  # Can be tuned
        )

# === STAGE 5: Build Final Train/Test Files Per User ===
if run_merge:
    print("\n=== Building Final Train/Test Files ===")
    os.makedirs('final_user_files', exist_ok=True)

    merge_selected_features(
        selected_folder='selected_features',
        output_folder='final_user_files',
        top_n=5  # Keep top 5 features per user
    )

    print("\n=== Creating Final Unified Train/Test Files ===")
    build_final_user_files(
        unified_folder='final_user_files',
        data_folder='selected_features',
        output_folder='final_user_csvs'
    )

print("\n🎉 Done.")
