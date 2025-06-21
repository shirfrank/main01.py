
Sleep Quality Prediction from Mobile Sensor Data
================================================

Overview:
---------
This project predicts sleep quality using behavioral and environmental features derived from smartphone sensor data.
It includes a full pipeline from raw sensor cleaning to per-user train/test file generation ready for modeling.

Main Pipeline Stages:
---------------------
1. Sensor Data Exploration:
   - explore_sensors.py: Logs types and volume of raw sensor data per session.

2. Data Cleaning:
   - clean.py: Interpolates or filters raw accelerometer, light, Wi-Fi, location, calls, and screen sensors.

3. Sleep Window Estimation:
   - get_sensor_window.py: Estimates daily sleep and awake windows for each user based on sensor heuristics.

4. Feature Extraction:
   - feature_extraction_v2.py: Extracts ~250 features per day from each user's sensor streams, including:
     • Accelerometer: movement ratios, still periods, entropy
     • Light: brightness patterns, peak time
     • Screen: usage patterns, screen-light overlap
     • Wi-Fi: signal consistency, AP changes
     • Calls: usage stats
     • Location: mobility, stillness
     • Circadian: misalignment, stimulation before sleep

5. Train/Test Split:
   - train_test.py: Generates up to 3 temporal splits per user using tail-end test sampling.

6. Feature Vetting:
   - vetting.py: Per-user feature normalization and selection using Spearman correlation and redundancy filtering.

7. Feature Selection:
   - feature_selection.py: Exhaustive wrapper search for top k features (by MSE using Ridge CV).

8. Merging and Final File Creation:
   - merge_features.py: Aggregates selected features per user and split.
   - build_final_user_files.py: Creates final CSV files per user for downstream modeling.

Output:
-------
• cleaned_*.csv: cleaned raw sensor data
• features_session_*.csv: daily extracted features per session
• train/test_user_uid_splitN.csv: train/test samples per user and split
• *_vetted.csv: vetted, normalized features
• selected_train/test_*.csv: selected features per split
• final_user_csvs/: final train/test datasets per user

Author:
-------
Yuval Berkovich and Shir Frank
Biomedical Engineering, Tel Aviv University
