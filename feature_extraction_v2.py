import pandas as pd
from numpy import exp, angle, pi
import numpy as np

from scipy.stats import entropy
from itertools import groupby


"""
Feature Extraction Module for Sleep Quality Prediction
=======================================================

This module defines functions to compute daily behavioral and environmental features 
from mobile sensor data (accelerometer, light, screen, calls, Wi-Fi, location) to predict sleep quality.

Each function extracts statistical or behavioral metrics from time windows aligned to a specific day, 
including subwindows such as 'day', 'pre_sleep', and 'sleep'.

Features include:
- Physical activity: movement ratios, still periods, acceleration entropy
- Light exposure: brightness stats, entropy, high-light durations
- Phone usage: screen session stats, gap before sleep, screen-light overlaps
- Calls: durations and counts per type
- Environmental context: Wi-Fi strength, mobility from location data
- Circadian metrics: screen/light usage rhythm and misalignment
- Composite behavioral scores: circadian_disruption_score, evening_stimulation_index,
  and digital_abstinence_before_sleep — designed to quantify habits impacting sleep quality.

All feature dictionaries are returned in flat format, suitable for direct use in machine learning pipelines.
"""
def compute_entropy(series):
    value_counts = series.value_counts(normalize=True, dropna=True)
    return entropy(value_counts)

def compute_accel_features(accel_window):
    features = {}

    if accel_window.empty:
        features.update({
            'accel_movement_ratio': 0,
            'accel_evening_movement_ratio': 0,
            'accel_night_movement_ratio': 0,
            'accel_morning_movement_ratio': 0,
            'accel_mean_mag': 0,
            'accel_std_mag': 0,
            'accel_entropy': 0
        })
    else:
        #  magnitude
        accel_window['magnitude'] = np.sqrt(accel_window['x']**2 + accel_window['y']**2 + accel_window['z']**2)
        accel_window['deviation'] = abs(accel_window['magnitude'] - 9.8)


        threshold = 0.1
        accel_window['is_movement'] = accel_window['deviation'] > threshold
        # --- Longest still period ---
        still_series = ~accel_window['is_movement'].values.astype(bool)


        still_runs = [sum(1 for _ in group) for key, group in groupby(still_series) if key == True]

        if still_runs:
            longest_still_in_minutes = max(still_runs) * 15  #there is a sample each 15 minutes
        else:
            longest_still_in_minutes = 0

        features['accel_longest_still_period'] = longest_still_in_minutes

        # how much movement for window
        features['accel_movement_ratio'] = accel_window['is_movement'].mean()

        # Time slicing
        accel_window['hour'] = accel_window['datetime'].dt.hour

        # evening
        evening_data = accel_window[accel_window['hour'].between(18, 21)]
        features['accel_evening_movement_ratio'] = evening_data['is_movement'].mean() if not evening_data.empty else 0

        # 22:00–05:00
        night_data = accel_window[(accel_window['hour'] >= 22) | (accel_window['hour'] <= 5)]
        features['accel_night_movement_ratio'] = night_data['is_movement'].mean() if not night_data.empty else 0

        # morning
        morning_data = accel_window[accel_window['hour'].between(6, 9)]
        features['accel_morning_movement_ratio'] = morning_data['is_movement'].mean() if not morning_data.empty else 0

        # average and std
        features['accel_mean_mag'] = accel_window['magnitude'].mean()
        features['accel_std_mag'] = accel_window['magnitude'].std()

        # Entropy magnitude
        features['accel_entropy'] = compute_entropy(pd.cut(accel_window['magnitude'], bins=10))

    return features

def compute_light_features(light_window, subwindows):
    features = {}
    if light_window.empty:
        for period in ['day', 'pre_sleep', 'sleep']:
            features[f'light_mean_{period}'] = 0
            features[f'light_max_{period}'] = 0
            features[f'light_entropy_{period}'] = 0
            features[f'light_high_duration_{period}'] = 0
        features['light_last_peak_time'] = -1
        return features

    light_window = light_window.copy()
    light_window['lux'] = pd.to_numeric(light_window['value'], errors='coerce').fillna(0)

    for period, (start, end) in subwindows.items():
        sub = light_window[(light_window['datetime'] >= start) & (light_window['datetime'] < end)]
        if sub.empty:
            features[f'light_mean_{period}'] = 0
            features[f'light_max_{period}'] = 0
            features[f'light_entropy_{period}'] = 0
            features[f'light_high_duration_{period}'] = 0
        else:
            features[f'light_mean_{period}'] = sub['lux'].mean()
            features[f'light_max_{period}'] = sub['lux'].max()
            features[f'light_entropy_{period}'] = compute_entropy(pd.cut(sub['lux'], bins=10))
            features[f'light_high_duration_{period}'] = (sub['lux'] > 100).mean()

    if light_window['lux'].max() > 0:
        peak_time = light_window.loc[light_window['lux'].idxmax(), 'datetime']
        features['light_last_peak_time'] = peak_time.hour + peak_time.minute / 60.0
    else:
        features['light_last_peak_time'] = -1
    return features

def compute_calls_features(calls_window, subwindows):
    features = {}
    if calls_window.empty:
        for period in ['day', 'pre_sleep', 'sleep']:
            features[f'calls_total_{period}'] = 0
            features[f'calls_duration_{period}'] = 0
        features.update({
            'calls_num_incoming': 0,
            'calls_num_outgoing': 0,
            'calls_num_missed': 0,
            'calls_last_call_time': -1,
            'calls_first_call_next_day': -1
        })
        return features

    calls_window = calls_window.copy()
    calls_window['hour'] = calls_window['datetime'].dt.hour
    calls_window['status'] = calls_window['sensor_status']
    calls_window['duration_sec'] = pd.to_numeric(calls_window['value'], errors='coerce').fillna(0)

    for period, (start, end) in subwindows.items():
        sub = calls_window[(calls_window['datetime'] >= start) & (calls_window['datetime'] < end)]
        features[f'calls_total_{period}'] = sub.shape[0]
        features[f'calls_duration_{period}'] = sub['duration_sec'].sum() / 3600  # hours

    features['calls_num_incoming'] = (calls_window['status'] == 'INCOMING').sum()
    features['calls_num_outgoing'] = (calls_window['status'] == 'OUTGOING').sum()
    features['calls_num_missed'] = (calls_window['status'] == 'MISSED').sum()

    if not calls_window.empty:
        last_call_time = calls_window['datetime'].max()
        features['calls_last_call_time'] = last_call_time.hour + last_call_time.minute / 60.0
        next_day_calls = calls_window[calls_window['hour'] >= 5]
        if not next_day_calls.empty:
            first_call = next_day_calls['datetime'].min()
            features['calls_first_call_next_day'] = first_call.hour + first_call.minute / 60.0
        else:
            features['calls_first_call_next_day'] = -1

    return features

# --------- Screen features ---------
def compute_screen_features(screen_window, light_window, subwindows):
    screen_window = screen_window.copy()

    features = {}

    if screen_window.empty:
        for period in ['day', 'pre_sleep', 'sleep']:
            features[f'screen_events_{period}'] = 0
            features[f'screen_inter_event_std_{period}'] = 0
            features[f'screen_total_duration_{period}'] = 0
            features[f'screen_avg_session_duration_{period}'] = 0
            features[f'screen_cluster_count_{period}'] = 0
            features[f'light_screen_overlap_ratio_{period}'] = 0

        features['screen_last_on_time'] = 0
        features['screen_first_on_time_day'] = -1
        features['screen_gap_before_sleep'] = -1
        return features

    screen_window['datetime'] = pd.to_datetime(screen_window['datetime'])
    screen_window = screen_window.sort_values('datetime')
    screen_window['inter_event_time'] = screen_window['datetime'].diff().dt.total_seconds()

    # Preprocess: pair up 'on' and 'off' events
    on_off_pairs = []
    current_on = None
    for _, row in screen_window.iterrows():
        if row['value'].lower() == 'on':
            current_on = row['datetime']
        elif row['value'].lower() == 'off' and current_on:
            on_off_pairs.append((current_on, row['datetime']))
            current_on = None

    for period, (start, end) in subwindows.items():
        # Filter on/off pairs within the current time window
        period_pairs = [
            (on_time, off_time) for on_time, off_time in on_off_pairs
            if start <= on_time < end and on_time < off_time
        ]

        session_durations = [(off - on).total_seconds() for on, off in period_pairs]
        total_duration_sec = sum(session_durations)
        total_duration_hr = total_duration_sec / 3600
        avg_duration = np.mean(session_durations) if session_durations else 0

        features[f'screen_events_{period}'] = len(period_pairs)
        features[f'screen_total_duration_{period}'] = total_duration_hr
        features[f'screen_avg_session_duration_{period}'] = avg_duration / 60  # in minutes

        # Inter-event std
        sub = screen_window[(screen_window['datetime'] >= start) & (screen_window['datetime'] < end)]
        inter_times = sub['datetime'].diff().dt.total_seconds()
        features[f'screen_inter_event_std_{period}'] = inter_times.std() if sub.shape[0] > 1 else 0

        # Cluster count (events <3 min apart)
        clusters = (inter_times < 180).sum() if inter_times.size > 1 else 0
        features[f'screen_cluster_count_{period}'] = clusters

        #  Light-Screen Overlap
        if not light_window.empty:
            light_sub = light_window[(light_window['datetime'] >= start) & (light_window['datetime'] < end)].copy()
            light_sub['lux'] = pd.to_numeric(light_sub['value'], errors='coerce').fillna(0)

            overlap_count = 0
            for on_time, _ in period_pairs:
                time_diff = (light_sub['datetime'] - on_time).abs()
                closest_idx = time_diff.idxmin() if not time_diff.empty else None
                if closest_idx is not None and time_diff[closest_idx].total_seconds() <= 300:
                    if light_sub.loc[closest_idx, 'lux'] > 100:
                        overlap_count += 1

            overlap_ratio = overlap_count / len(period_pairs) if len(period_pairs) > 0 else 0
        else:
            overlap_ratio = 0

        features[f'light_screen_overlap_ratio_{period}'] = overlap_ratio

    # Last screen-on time
    last_on = screen_window[screen_window['value'].str.lower() == 'on']
    if not last_on.empty:
        last_time = last_on['datetime'].max()
        features['screen_last_on_time'] = last_time.hour + last_time.minute / 60.0
    else:
        features['screen_last_on_time'] = 0

    # First screen-on in "day" window
    day_window = subwindows.get("day", (None, None))
    first_day = screen_window[(screen_window['value'].str.lower() == 'on') &
                              (screen_window['datetime'] >= day_window[0]) & (screen_window['datetime'] < day_window[1])]
    if not first_day.empty:
        first_time = first_day['datetime'].min()
        features['screen_first_on_time_day'] = first_time.hour + first_time.minute / 60.0
    else:
        features['screen_first_on_time_day'] = -1

    # Gap to sleep
    if not last_on.empty and 'sleep' in subwindows:
        sleep_start = subwindows['sleep'][0]
        features['screen_gap_before_sleep'] = (sleep_start - last_time).total_seconds() / 60.0 if sleep_start > last_time else 0
    else:
        features['screen_gap_before_sleep'] = -1

    return features



# --------- Wi-Fi features ---------
def compute_wifi_features(wifi_window, subwindows):
    features = {}

    for period, (start, end) in subwindows.items():
        sub = wifi_window[(wifi_window['datetime'] >= start) & (wifi_window['datetime'] < end)].copy()

        features[f'wifi_scans_{period}'] = sub['datetime'].nunique()
        features[f'wifi_unique_aps_{period}'] = sub['suuid'].nunique() if 'suuid' in sub.columns else 0

        if not sub.empty and 'suuid' in sub.columns and 'level' in sub.columns:
            sub['level'] = pd.to_numeric(sub['level'], errors='coerce')

            # Signal stats
            features[f'wifi_mean_signal_{period}'] = sub['level'].mean()
            features[f'wifi_signal_std_{period}'] = sub['level'].std()

            # Dominant AP ratio
            ap_counts = sub['suuid'].value_counts(normalize=True)
            features[f'wifi_dominant_ap_ratio_{period}'] = ap_counts.max() if not ap_counts.empty else 0

            # Entropy of APs
            features[f'wifi_entropy_{period}'] = entropy(ap_counts) if len(ap_counts) > 1 else 0

            # Strong APs
            features[f'wifi_strong_ap_ratio_{period}'] = (sub['level'] > -65).mean()

            # === NEW: Average number of APs per scan ===
            aps_per_scan = sub.groupby('datetime')['suuid'].nunique()
            features[f'wifi_avg_num_aps_per_scan_{period}'] = aps_per_scan.mean() if not aps_per_scan.empty else 0

            # === NEW: Number of dominant AP changes ===
            dominant_ap_per_scan = sub.groupby('datetime')['suuid'].agg(lambda x: x.value_counts().idxmax())
            dominant_changes = (dominant_ap_per_scan != dominant_ap_per_scan.shift()).sum() - 1
            features[f'wifi_dominant_ap_changes_{period}'] = max(0, dominant_changes)
        else:
            features[f'wifi_mean_signal_{period}'] = 0
            features[f'wifi_signal_std_{period}'] = 0
            features[f'wifi_dominant_ap_ratio_{period}'] = 0
            features[f'wifi_entropy_{period}'] = 0
            features[f'wifi_strong_ap_ratio_{period}'] = 0
            features[f'wifi_avg_num_aps_per_scan_{period}'] = 0
            features[f'wifi_dominant_ap_changes_{period}'] = 0

    return features


# --------- Location features ---------
import json
import pandas as pd

def compute_location_features(location_window, subwindows):
    features = {}

    if location_window.empty:
        for period in ['day', 'pre_sleep', 'sleep']:
            features[f'location_mobility_duration_{period}'] = 0
            features[f'location_immobility_duration_{period}'] = 0
            features[f'location_mobility_ratio_{period}'] = 0
            features[f'location_max_still_duration_{period}'] = 0
            features[f'location_num_transitions_{period}'] = 0
            features[f'location_avg_speed_{period}'] = 0
        return features

    location_window = location_window.copy()
    location_window = location_window.dropna(subset=['datetime', 'data'])
    location_window['parsed'] = location_window['data'].apply(json.loads)
    location_window['distance'] = location_window['parsed'].apply(lambda d: d.get('distance', 0))
    location_window['speed'] = location_window['parsed'].apply(lambda d: d.get('speed', 0))

    location_window = location_window.sort_values('datetime')
    location_window['time_diff'] = location_window['datetime'].diff().dt.total_seconds().fillna(0)
    location_window['is_mobile'] = location_window['distance'] > 3.5

    for period, (start, end) in subwindows.items():
        sub = location_window[(location_window['datetime'] >= start) & (location_window['datetime'] < end)]

        if not sub.empty:
            # === Original durations ===
            mobile_duration = sub[sub['is_mobile']]['time_diff'].sum() / 3600  # in hours
            immobile_duration = sub[~sub['is_mobile']]['time_diff'].sum() / 3600
            total_time = mobile_duration + immobile_duration
            mobility_ratio = mobile_duration / total_time if total_time > 0 else 0

            features[f'location_mobility_duration_{period}'] = mobile_duration
            features[f'location_immobility_duration_{period}'] = immobile_duration
            features[f'location_mobility_ratio_{period}'] = mobility_ratio

            # === NEW: Longest still period ===
            still_series = ~sub['is_mobile'].values
            runs = [sum(1 for _ in group) for val, group in groupby(still_series) if val]
            max_still = max(runs) * sub['time_diff'].median() / 60 if runs else 0  # in minutes
            features[f'location_max_still_duration_{period}'] = max_still

            # === NEW: Number of transitions (mobile <-> still) ===
            transitions = (sub['is_mobile'] != sub['is_mobile'].shift()).sum() - 1
            features[f'location_num_transitions_{period}'] = max(0, transitions)

            # === NEW: Average speed during movement ===
            moving = sub[sub['is_mobile']]
            avg_speed = moving['speed'].mean() if not moving.empty else 0
            features[f'location_avg_speed_{period}'] = avg_speed
        else:
            features[f'location_mobility_duration_{period}'] = 0
            features[f'location_immobility_duration_{period}'] = 0
            features[f'location_mobility_ratio_{period}'] = 0
            features[f'location_max_still_duration_{period}'] = 0
            features[f'location_num_transitions_{period}'] = 0
            features[f'location_avg_speed_{period}'] = 0

    return features

def compute_circadian_features(screen_window, light_window, accel_window, subwindows):
    features = {}

    # --- Helper: circular mean hour ---
    def circadian_center(df):
        if df.empty:
            return -1
        hours = df['datetime'].dt.hour + df['datetime'].dt.minute / 60
        radians = 2 * pi * hours / 24
        vector = exp(1j * radians)
        center = angle(vector.mean())
        return (24 * center / (2 * pi)) % 24

    # --- 1. Circular centers ---
    features['circadian_screen_center'] = circadian_center(screen_window)
    features['circadian_light_center'] = circadian_center(light_window)

    # --- 2. Misalignment from ideal bedtime (22:00) ---
    for name in ['screen', 'light']:
        center = features[f'circadian_{name}_center']
        features[f'circadian_{name}_misalignment'] = abs(center - 22) if center >= 0 else -1

    # --- 3. Activity ratio during pre-sleep and sleep ---
    for name, win in [('screen', screen_window), ('light', light_window)]:
        total = win.shape[0]
        pre_sleep = win[(win['datetime'] >= subwindows['pre_sleep'][0]) & (win['datetime'] < subwindows['pre_sleep'][1])].shape[0]
        sleep = win[(win['datetime'] >= subwindows['sleep'][0]) & (win['datetime'] < subwindows['sleep'][1])].shape[0]
        features[f'{name}_pre_sleep_ratio'] = pre_sleep / total if total else 0
        features[f'{name}_sleep_ratio'] = sleep / total if total else 0

    # --- 4. Screen wind-down slope ---
    counts = [
        screen_window[(screen_window['datetime'] >= subwindows[w][0]) & (screen_window['datetime'] < subwindows[w][1])].shape[0]
        for w in ['day', 'pre_sleep', 'sleep']
    ]
    features['screen_wind_down_slope'] = counts[0] - counts[2]

    # --- 5. Light peak hour per window ---
    for w in ['day', 'pre_sleep', 'sleep']:
        sub = light_window[(light_window['datetime'] >= subwindows[w][0]) & (light_window['datetime'] < subwindows[w][1])]
        if not sub.empty:
            peak = sub.loc[sub['value'].astype(float).idxmax(), 'datetime']
            features[f'light_peak_hour_{w}'] = peak.hour + peak.minute / 60.0
        else:
            features[f'light_peak_hour_{w}'] = -1

    # --- 6. Acceleration movement variability over the day ---
    if not accel_window.empty and 'is_movement' in accel_window.columns:
        accel_window = accel_window.copy()
        accel_window['hour'] = accel_window['datetime'].dt.hour
        hourly_movement = accel_window.groupby('hour')['is_movement'].mean()
        features['accel_activity_hourly_std'] = hourly_movement.std()
    else:
        features['accel_activity_hourly_std'] = 0

    # --- 7. Composite circadian + behavioral features ---
    features['circadian_disruption_score'] = (
        features.get('circadian_screen_misalignment', 0) +
        features.get('circadian_light_misalignment', 0) +
        features.get('screen_sleep_ratio', 0) * 10 +
        features.get('light_sleep_ratio', 0) * 10
    )

    features['evening_stimulation_index'] = (
        features.get('screen_events_pre_sleep', 0) +
        features.get('light_high_duration_pre_sleep', 0) * 10 +
        features.get('calls_total_pre_sleep', 0)
    )

    features['digital_abstinence_before_sleep'] = features.get('screen_gap_before_sleep', -1)

    return features
