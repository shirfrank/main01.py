import pandas as pd
from numpy import exp, angle, pi
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import entropy
from itertools import groupby
import json


def compute_entropy(series):
    value_counts = series.value_counts(normalize=True, dropna=True)
    return entropy(value_counts)

def butter_highpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def compute_accel_features(accel_window, subwindows):
    features = {}
    accel_window = accel_window.dropna(subset=['x', 'y', 'z'])

    if accel_window.empty:
        for period in ['day', 'pre_sleep', 'sleep']:
            features[f'accel_movement_ratio_{period}'] = 0
            features[f'accel_mean_mag_{period}'] = 0
            features[f'accel_std_mag_{period}'] = 0
            features[f'accel_entropy_{period}'] = 0
        features['accel_longest_still_period'] = 0
        return features

    accel_window = accel_window.copy()

    # --- Apply high-pass filter to each axis ---
    fs = 1 / (15 * 60)  # Sampling rate: 1 sample per 15 minutes
    cutoff = 0.25  # Hz

    for axis in ['x', 'y', 'z']:
        try:
            accel_window[f'{axis}_hp'] = butter_highpass_filter(accel_window[axis], cutoff, fs)
        except ValueError:
            accel_window[f'{axis}_hp'] = 0  # fallback in case of insufficient samples

    accel_window['magnitude_hp'] = np.sqrt(
        accel_window['x_hp']**2 + accel_window['y_hp']**2 + accel_window['z_hp']**2
    )

    accel_window['is_movement'] = accel_window['magnitude_hp'] > 0.05  # small threshold

    # Longest still period
    still_series = ~accel_window['is_movement'].values
    still_runs = [sum(1 for _ in group) for key, group in groupby(still_series) if key]
    features['accel_longest_still_period'] = (max(still_runs) * 15) / 60  # converted to hours

    for period, (start, end) in subwindows.items():
        sub = accel_window[(accel_window['datetime'] >= start) & (accel_window['datetime'] < end)]

        if sub.empty:
            features[f'accel_movement_ratio_{period}'] = 0
            features[f'accel_mean_mag_{period}'] = 0
            features[f'accel_std_mag_{period}'] = 0
            features[f'accel_entropy_{period}'] = 0
        else:
            features[f'accel_movement_ratio_{period}'] = sub['is_movement'].mean()
            features[f'accel_mean_mag_{period}'] = sub['magnitude_hp'].mean()
            features[f'accel_std_mag_{period}'] = sub['magnitude_hp'].std()
            features[f'accel_entropy_{period}'] = compute_entropy(pd.cut(sub['magnitude_hp'], bins=10))

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
def compute_screen_features(screen_window, subwindows):
    screen_window = screen_window.copy()

    # Filter to only "on" events
    screen_window = screen_window[screen_window['value'].str.lower() == 'on']
    features = {}

    if screen_window.empty:
        for period in ['day', 'pre_sleep', 'sleep']:
            features[f'screen_events_{period}'] = 0
            features[f'screen_inter_event_std_{period}'] = 0
            features[f'screen_total_duration_{period}'] = 0
            features[f'screen_avg_session_duration_{period}'] = 0
            features[f'screen_cluster_count_{period}'] = 0

        features['screen_last_on_time'] = 0
        features['screen_first_on_time_day'] = -1
        features['screen_gap_before_sleep'] = -1
        return features

    screen_window = screen_window.copy()
    screen_window = screen_window.sort_values('datetime')
    screen_window['inter_event_time'] = screen_window['datetime'].diff().dt.total_seconds()

    for period, (start, end) in subwindows.items():
        sub = screen_window[(screen_window['datetime'] >= start) & (screen_window['datetime'] < end)]
        inter_times = sub['datetime'].diff().dt.total_seconds()

        features[f'screen_events_{period}'] = sub.shape[0]
        features[f'screen_inter_event_std_{period}'] = inter_times.std() if sub.shape[0] > 1 else 0

        # Screen session duration assumptions
        assumed_duration = 60  # seconds per screen-on event
        total_duration = (sub.shape[0] * assumed_duration) / 3600  # in hours
        avg_duration = assumed_duration if sub.shape[0] > 0 else 0

        features[f'screen_total_duration_{period}'] = total_duration
        features[f'screen_avg_session_duration_{period}'] = avg_duration

        # Cluster count (events less than 3 minutes apart)
        clusters = (inter_times < 180).sum() if sub.shape[0] > 1 else 0
        features[f'screen_cluster_count_{period}'] = clusters

    # Time of last screen-on event
    last_time = screen_window['datetime'].max()
    features['screen_last_on_time'] = last_time.hour + last_time.minute / 60.0

    # First screen-on time in the day window
    day_window = subwindows.get("day", (None, None))
    first_day = screen_window[(screen_window['datetime'] >= day_window[0]) & (screen_window['datetime'] < day_window[1])]
    if not first_day.empty:
        first_time = first_day['datetime'].min()
        features['screen_first_on_time_day'] = first_time.hour + first_time.minute / 60.0
    else:
        features['screen_first_on_time_day'] = -1

    # Gap between last screen use and start of sleep window
    sleep_start = subwindows['sleep'][0]
    features['screen_gap_before_sleep'] = (sleep_start - last_time).total_seconds() / 60.0 if sleep_start > last_time else 0

    return features

# --------- Wi-Fi features ---------
def compute_wifi_features(wifi_window, subwindows):
    features = {}

    for period, (start, end) in subwindows.items():
        sub = wifi_window[(wifi_window['datetime'] >= start) & (wifi_window['datetime'] < end)]

        # Basic metrics
        features[f'wifi_scans_{period}'] = sub.shape[0]
        features[f'wifi_unique_aps_{period}'] = sub['suuid'].nunique() if 'suuid' in sub.columns else 0

        if not sub.empty and 'suuid' in sub.columns and 'level' in sub.columns:
            # Convert signal strength to numeric (RSSI in dB)
            sub = sub.copy()
            sub.loc[:, 'level'] = pd.to_numeric(sub['level'], errors='coerce')

            # Signal strength features
            features[f'wifi_mean_signal_{period}'] = sub['level'].mean()
            features[f'wifi_signal_std_{period}'] = sub['level'].std()

            # Dominance: strongest visible AP over time
            ap_counts = sub['suuid'].value_counts(normalize=True)
            features[f'wifi_dominant_ap_ratio_{period}'] = ap_counts.max() if not ap_counts.empty else 0

            # Entropy of seen APs — high entropy = noisy public area, low = stable environment
            features[f'wifi_entropy_{period}'] = entropy(ap_counts) if len(ap_counts) > 1 else 0

            # Strong AP ratio: % of APs with signal stronger than -65 dB
            features[f'wifi_strong_ap_ratio_{period}'] = (sub['level'] > -65).mean()
        else:
            features[f'wifi_mean_signal_{period}'] = 0
            features[f'wifi_signal_std_{period}'] = 0
            features[f'wifi_dominant_ap_ratio_{period}'] = 0
            features[f'wifi_entropy_{period}'] = 0
            features[f'wifi_strong_ap_ratio_{period}'] = 0

    return features


# --------- Location features ---------
def compute_location_features(location_window, subwindows):
    features = {}

    if location_window.empty:
        for period in ['day', 'pre_sleep', 'sleep']:
            features[f'location_mobility_duration_{period}'] = 0
            features[f'location_immobility_duration_{period}'] = 0
            features[f'location_mobility_ratio_{period}'] = 0
        return features

    location_window = location_window.copy()
    location_window['value'] = pd.to_numeric(location_window['value'], errors='coerce')
    location_window = location_window.dropna(subset=['value', 'datetime'])
    location_window = location_window.sort_values('datetime')

    location_window['time_diff'] = location_window['datetime'].diff().dt.total_seconds().fillna(0)
    location_window['is_mobile'] = location_window['value'] > 3.5  # movement threshold in meters

    for period, (start, end) in subwindows.items():
        sub = location_window[(location_window['datetime'] >= start) & (location_window['datetime'] < end)]

        if not sub.empty:
            mobile_duration = sub[sub['is_mobile']]['time_diff'].sum() / 3600  # hours
            immobile_duration = sub[~sub['is_mobile']]['time_diff'].sum() / 3600
            total_time = mobile_duration + immobile_duration
            mobility_ratio = mobile_duration / total_time if total_time > 0 else 0

            features[f'location_mobility_duration_{period}'] = mobile_duration
            features[f'location_immobility_duration_{period}'] = immobile_duration
            features[f'location_mobility_ratio_{period}'] = mobility_ratio
        else:
            features[f'location_mobility_duration_{period}'] = 0
            features[f'location_immobility_duration_{period}'] = 0
            features[f'location_mobility_ratio_{period}'] = 0

    return features
def compute_circadian_features(screen_window, light_window, accel_window, subwindows):
    features = {}

    # --- Helper: circular mean hour ---
    def circadian_center(df):
        if df.empty: return -1
        hours = df['datetime'].dt.hour + df['datetime'].dt.minute / 60
        radians = 2 * pi * hours / 24
        vector = exp(1j * radians)
        center = angle(vector.mean())
        return (24 * center / (2 * pi)) % 24

    # --- 1. Circular centers ---
    features['circadian_screen_center'] = circadian_center(screen_window)
    features['circadian_light_center'] = circadian_center(light_window)

    # Misalignment from ideal bedtime (22:00)
    for name in ['screen', 'light']:
        center = features[f'circadian_{name}_center']
        features[f'circadian_{name}_misalignment'] = abs(center - 22) if center >= 0 else -1

    # --- 2. Pre-sleep & sleep activity ratio ---
    for name, win in [('screen', screen_window), ('light', light_window)]:
        total = win.shape[0]
        pre_sleep = win[(win['datetime'] >= subwindows['pre_sleep'][0]) & (win['datetime'] < subwindows['pre_sleep'][1])].shape[0]
        sleep = win[(win['datetime'] >= subwindows['sleep'][0]) & (win['datetime'] < subwindows['sleep'][1])].shape[0]
        features[f'{name}_pre_sleep_ratio'] = pre_sleep / total if total else 0
        features[f'{name}_sleep_ratio'] = sleep / total if total else 0

    # --- 3. Screen wind-down slope ---
    counts = [
        screen_window[(screen_window['datetime'] >= subwindows[w][0]) & (screen_window['datetime'] < subwindows[w][1])].shape[0]
        for w in ['day', 'pre_sleep', 'sleep']
    ]
    features['screen_wind_down_slope'] = counts[0] - counts[2]

    # --- 4. Light peak hour per window ---
    for w in ['day', 'pre_sleep', 'sleep']:
        sub = light_window[(light_window['datetime'] >= subwindows[w][0]) & (light_window['datetime'] < subwindows[w][1])]
        if not sub.empty:
            peak = sub.loc[sub['value'].astype(float).idxmax(), 'datetime']
            features[f'light_peak_hour_{w}'] = peak.hour + peak.minute / 60.0
        else:
            features[f'light_peak_hour_{w}'] = -1

    # --- 5. Accel activity std over hours ---
    if not accel_window.empty and 'is_movement' in accel_window.columns:
        accel_window = accel_window.copy()
        accel_window['hour'] = accel_window['datetime'].dt.hour
        hourly_movement = accel_window.groupby('hour')['is_movement'].mean()
        features['accel_activity_hourly_std'] = hourly_movement.std()
    else:
        features['accel_activity_hourly_std'] = 0

    return features