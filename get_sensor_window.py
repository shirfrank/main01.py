import pandas as pd
import numpy as np
from datetime import timedelta

def estimate_sleep_wake_times(screen_df, light_df, accel_df, location_df, wifi_df, label_df, uid, current_date_str):
    current_date = pd.to_datetime(current_date_str).normalize()
    night_start = current_date - timedelta(hours=6)  # from 18:00 the day before
    morning_end = current_date + timedelta(hours=12)  # until 12:00 same day

    def filter_user_time(df):
        if isinstance(df, tuple):
            if len(df) > 0 and isinstance(df[0], pd.DataFrame):
                df = df[0]
            else:
                return pd.DataFrame()
        if not isinstance(df, pd.DataFrame) or 'uid' not in df.columns or 'datetime' not in df.columns:
            return pd.DataFrame()
        return df[(df['uid'] == uid) & (df['datetime'] >= night_start) & (df['datetime'] <= morning_end)].copy()

    screen = filter_user_time(screen_df)
    light = filter_user_time(light_df)
    accel = filter_user_time(accel_df)
    location = filter_user_time(location_df)
    wifi = filter_user_time(wifi_df)

    time_index = pd.date_range(night_start, morning_end, freq='5min')
    score_df = pd.DataFrame(index=time_index)

    if not screen.empty:
        # Convert 'value' to binary: 1 if 'off', 0 if 'on'
        screen['inactive'] = (screen['value'].str.lower() == 'off').astype(int)

        # Resample on 'inactive' only
        screen.set_index('datetime', inplace=True)
        screen_score = screen[['inactive']].resample('5min').mean()['inactive'].reindex(time_index, fill_value=0)

        # Higher score = more inactivity = more likely to be asleep
        score_df['screen'] = screen_score

    if not light.empty:
        light['value'] = pd.to_numeric(light['value'], errors='coerce')
        light_score = light.set_index('datetime').resample('5min')['value'].mean().reindex(time_index, fill_value=0)
        score_df['light'] = (light_score < 15).astype(float)

    if not accel.empty:
        accel['mag'] = np.sqrt(accel['x']**2 + accel['y']**2 + accel['z']**2)
        accel_std = accel.set_index('datetime').resample('5min')['mag'].std().reindex(time_index, fill_value=0)
        score_df['accel'] = (accel_std < 0.1).astype(float)

    if not location.empty:
        # Ensure 'value' is numeric — it represents distance
        location['value'] = pd.to_numeric(location['value'], errors='coerce')

        # Count number of unique locations in each 5-min bin using encrypted location ID 'x'
        location_changes = location.set_index('datetime').resample('5min')['x'].nunique().reindex(time_index,
                                                                                                  fill_value=0)
        # Compute average movement per 5-min
        avg_movement = location.set_index('datetime').resample('5min')['value'].mean().reindex(time_index,
                                                                                              fill_value=np.nan)
        # Score 1 if both movement is low and location didn't change much
        score_df['location'] = ((avg_movement < 3.5) & (location_changes < 4)).astype(float)

    if not wifi.empty and 'suuid' in wifi.columns and 'level' in wifi.columns:
        wifi['level'] = pd.to_numeric(wifi['level'], errors='coerce')
        wifi_filtered = wifi.dropna(subset=['level'])

        # Stability: dominant suuid ratio
        wifi_stability = wifi_filtered.set_index('datetime').resample('5min')['suuid'].agg(
            lambda x: x.value_counts(normalize=True).max()
        ).reindex(time_index, fill_value=0)

        # Strength: median signal level (more negative = weaker signal)
        wifi_strength = wifi_filtered.set_index('datetime').resample('5min')['level'].median().reindex(time_index,
                                                                                                       fill_value=np.nan)

        # Score 1 if stable + reasonably strong signal (e.g., stronger than -75 dBm)
        wifi_score = ((wifi_stability > 0.8) & (wifi_strength > -75)).astype(float)
        score_df['wifi'] = wifi_score

    score_df['total_score'] = score_df.mean(axis=1)
    print("[DEBUG] Sleep score percentiles:")
    print(score_df['total_score'].describe())
    sleep_periods = score_df['total_score'] >= 0.6
    candidates = []
    current_start = None
    for timestamp, is_sleep in sleep_periods.items():
        if is_sleep and current_start is None:
            current_start = timestamp
        elif not is_sleep and current_start is not None:
            duration = (timestamp - current_start).total_seconds() / 60
            if duration >= 60:
                candidates.append({'start': current_start, 'end': timestamp, 'duration': duration})
            current_start = None
    if current_start is not None:
        duration = (sleep_periods.index[-1] - current_start).total_seconds() / 60
        candidates.append({'start': current_start, 'end': sleep_periods.index[-1], 'duration': duration})

    if not candidates:
        print(f"⚠️ No sleep window estimated for UID {uid} on {current_date.date()}")
        return None, None

    best_segment = max(candidates, key=lambda x: x['duration'])
    sleep_start = best_segment['start']
    wake_time = best_segment['end']

    # Fallback for first session day
    default_start_time = pd.to_datetime(f"{current_date.date()} 04:00")
    try:
        user_labels = label_df[label_df['uid'] == uid].sort_values('Timestamp')
        current_row_idx = user_labels[user_labels['Timestamp'].dt.date == current_date.date()].index[0]
        prev_row_idx = user_labels.index.get_loc(current_row_idx) - 1
        if prev_row_idx >= 0:
            prev_row = user_labels.iloc[prev_row_idx]
            prev_wake_str = prev_row.iloc[8]
            prev_date = prev_row['Timestamp'].date()
            start_time = pd.to_datetime(f"{prev_date} {prev_wake_str}") if pd.notnull(prev_wake_str) else default_start_time
        else:
            start_time = default_start_time
    except Exception as e:
        print(f"⚠️ Could not determine prior wake time for UID {uid}: {e}")
        start_time = default_start_time

    pre_sleep_start = sleep_start - timedelta(hours=3)
    sub_windows = {
        'full_window': (start_time, wake_time),
        'day': (start_time, pre_sleep_start),
        'pre_sleep': (pre_sleep_start, sleep_start),
        'sleep': (sleep_start, wake_time)
    }

    try:
        label_row = label_df[
            (label_df['uid'] == uid) &
            (pd.to_datetime(label_df['Timestamp']).dt.date == current_date.date())
        ].iloc[0]

        # Access reported values from columns D (index 3) and I (index 8)
        reported_sleep_str = label_row.iloc[3]
        reported_wake_str = label_row.iloc[8]

        reported_sleep = None
        if pd.notnull(reported_sleep_str):
            reported_sleep_time = pd.to_datetime(reported_sleep_str).time()
            # ⏰ If sleep is before midnight, it happened the night before
            if reported_sleep_time.hour >= 0 and reported_sleep_time.hour < 12:
                sleep_date = current_date.date()
            else:
                sleep_date = current_date.date() - timedelta(days=1)
            reported_sleep = pd.to_datetime(f"{sleep_date} {reported_sleep_str}")

        # 🌅 Wake is always same-day
        if pd.notnull(reported_wake_str):
            reported_wake = pd.to_datetime(f"{current_date.date()} {reported_wake_str}")
        else:
            reported_wake = None

        diff_sleep = abs((sleep_start - reported_sleep).total_seconds()) / 3600 if reported_sleep else None
        diff_wake = abs((wake_time - reported_wake).total_seconds()) / 3600 if reported_wake else None

        print(f"\n🕵️ UID {uid} on {current_date.date()}")
        print(f"🛌 Reported sleep time: {reported_sleep.strftime('%Y-%m-%d %H:%M') if reported_sleep else 'N/A'}")
        print(f"🌅 Reported wake time:  {reported_wake.strftime('%Y-%m-%d %H:%M') if reported_wake else 'N/A'}")
        print(f"🧠 Estimated subwindows:")
        for label, (start, end) in sub_windows.items():
            print(f"  - {label:<10}: {start.strftime('%Y-%m-%d %H:%M')} ➜ {end.strftime('%Y-%m-%d %H:%M')}")
        if diff_sleep is not None and diff_wake is not None:
            print(f"📏 Difference from reported:")
            print(f"   Sleep time diff: {diff_sleep:.2f}h")
            print(f"   Wake time diff : {diff_wake:.2f}h")

        evaluation = {
            'uid': uid,
            'date': current_date.date(),
            'reported_sleep': reported_sleep,
            'estimated_sleep': sleep_start,
            'diff_sleep_hours': round(diff_sleep, 2) if diff_sleep is not None else None,
            'reported_wake': reported_wake,
            'estimated_wake': wake_time,
            'diff_wake_hours': round(diff_wake, 2) if diff_wake is not None else None,
        }

    except Exception as e:
        print(f"⚠️ Could not parse reported sleep/wake times for UID {uid}: {e}")
        evaluation = None

    return sub_windows, evaluation
