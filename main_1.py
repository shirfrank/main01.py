import os
import pandas as pd

# === External Modules ===
from explore_sensors import explore_sensors
from get_sensor_window import estimate_sleep_wake_times
from clean import clean_accel, clean_light, clean_screen, clean_wifi, clean_location, clean_calls
from feature_extraction_v2 import (
    compute_accel_features, compute_light_features, compute_screen_features,
    compute_wifi_features, compute_location_features, compute_calls_features,
    compute_circadian_features
)
from train_test import generate_multiple_splits_per_user
from vetting import vet_features_spearman_per_user
from feature_selection import exhaustive_feature_selection
from merge_features import merge_selected_features
from build_final_user_files import build_final_user_files

# === Prompt User for Stages to Run ===
print("Which stage(s) would you like to run?")
run_clean_extract = input("1. Clean & Extract Features? (y/n): ").lower() == 'y'
run_train = input("2. Train/Test Split? (y/n): ").lower() == 'y'
run_vetting = input("3. Feature Vetting? (y/n): ").lower() == 'y'
run_selection = input("4. Feature Selection? (y/n): ").lower() == 'y'
run_merge = input("5. Merge Final Files? (y/n): ").lower() == 'y'

# === File Paths ===
data_folder = 'data'
sessions = [
    {'name': 'A', 'sensor_path': os.path.join(data_folder, 'Session A', 'bhq_hisha_2025.xlsx'),
     'label_path': os.path.join(data_folder, 'Session A', 'Session_A_Label.csv')},
    {'name': 'B', 'sensor_path': os.path.join(data_folder, 'Session B', 'bhq_hisha_2025_s2.xlsx'),
     'label_path': os.path.join(data_folder, 'Session B', 'Session_B_Label.csv')},
    {'name': 'C', 'sensor_path': os.path.join(data_folder, 'Session C', 'bhq_hisha_2025_s3.xlsx'),
     'label_path': os.path.join(data_folder, 'Session C', 'Session_C_Label.csv')}
]

def extract_from_tuple(obj):
    return obj[0] if isinstance(obj, tuple) else obj

# === Stage 1: Clean & Extract Features ===
evaluations = []
if run_clean_extract:
    for session in sessions:
        name = session['name']
        print(f"\n=== Processing Session {name} ===")

        sensor_df = pd.read_excel(session['sensor_path'])
        label_df = pd.read_csv(session['label_path'])

        # Fix timestamp format
        label_df['Timestamp'] = pd.to_datetime(
            label_df['Timestamp'],
            format='%m/%d/%Y' if name in ['A', 'C'] else None
        )

        explore_sensors(sensor_df, f"Session {name}")
        sensor_df['datetime'] = pd.to_datetime(sensor_df['datetime'])
        sensor_df['date'] = sensor_df['datetime'].dt.date

        # Clean sensor data
        accel_df = extract_from_tuple(clean_accel(sensor_df))
        light_df = extract_from_tuple(clean_light(sensor_df))
        screen_df = extract_from_tuple(clean_screen(sensor_df))
        wifi_df = extract_from_tuple(clean_wifi(sensor_df))
        location_df = extract_from_tuple(clean_location(sensor_df))
        calls_df = extract_from_tuple(clean_calls(sensor_df))

        features_list = []

        for date in label_df['Timestamp'].dt.date.unique():
            for _, row in label_df[label_df['Timestamp'].dt.date == date].iterrows():
                uid = row['uid']
                label_date = row['Timestamp']
                sleep_score = row['Rate your overall sleep last night:']

                print(f"\nProcessing UID {uid} on {date}")

                windows, evaluation = estimate_sleep_wake_times(
                    screen_df, light_df, accel_df, location_df, wifi_df, label_df, uid, label_date
                )

                if windows is None:
                    continue
                if evaluation:
                    evaluations.append(evaluation)

                start, end = windows['full_window']
                filter_df = lambda df: df[(df['uid'] == uid) & (df['datetime'] >= start) & (df['datetime'] <= end)]

                row_features = {
                    'uid': uid,
                    'label_date': label_date,
                    'sleep_score': sleep_score
                }

                extractors = [
                    ("Accel", compute_accel_features, [filter_df(accel_df), windows]),
                    ("Light", compute_light_features, [filter_df(light_df), windows]),
                    ("Screen", compute_screen_features, [filter_df(screen_df), windows]),
                    ("WiFi", compute_wifi_features, [filter_df(wifi_df), windows]),
                    ("Location", compute_location_features, [filter_df(location_df), windows]),
                    ("Calls", compute_calls_features, [filter_df(calls_df), windows]),
                    ("Circadian", compute_circadian_features, [filter_df(screen_df), filter_df(light_df), filter_df(accel_df), windows])
                ]

                for name, func, args in extractors:
                    try:
                        row_features.update(func(*args))
                        print(f"✅ {name} features computed")
                    except Exception as e:
                        print(f"❌ Error in {name} features: {e}")

                features_list.append(row_features)

        features_df = pd.DataFrame(features_list)
        output_path = f"features_session_{name}.csv"
        features_df.to_csv(output_path, index=False)
        print(f"✅ Features saved to {output_path}")

    if evaluations:
        eval_df = pd.DataFrame(evaluations)
        eval_df.to_csv("sleep_estimation_differences.csv", index=False)
        valid = eval_df.dropna(subset=['diff_sleep_hours', 'diff_wake_hours'])
        print("\n📊 Sleep/Wake Difference Summary:")
        print(f"⏰ Avg sleep time diff: {valid['diff_sleep_hours'].mean():.2f} hours")
        print(f"🌅 Avg wake time diff: {valid['diff_wake_hours'].mean():.2f} hours")

# === Stage 2: Train/Test Split ===
if run_train:
    print("\n=== Running Train/Test Split ===")
    generate_multiple_splits_per_user()

# === Stage 3: Feature Vetting ===
if run_vetting:
    print("\n=== Running Feature Vetting ===")
    vet_features_spearman_per_user()

# === Stage 4: Feature Selection ===
if run_selection:
    print("\n=== Running Wrapper Feature Selection ===")
    selected_dir = 'selected_features'
    os.makedirs(selected_dir, exist_ok=True)

    existing_files = [
        f for f in os.listdir(selected_dir)
        if f.startswith("selected_train_user_") and f.endswith(".csv")
    ]

    if existing_files:
        print(f"📁 Found {len(existing_files)} selected feature files — skipping selection.")
    else:
        exhaustive_feature_selection(input_folder='.', output_folder=selected_dir, max_features=15)

# === Stage 5: Merge Final Files ===
if run_merge:
    print("\n=== Merging Final Train/Test Files ===")
    os.makedirs('final_user_files', exist_ok=True)

    merge_selected_features(
        selected_folder='selected_features',
        output_folder='final_user_files',
        top_n=5
    )

    build_final_user_files(
        unified_folder='final_user_files',
        data_folder='selected_features',
        output_folder='final_user_csvs'
    )

print("\n🎉 Done.")
