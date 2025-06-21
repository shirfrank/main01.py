import pandas as pd
import numpy as np


# ========================
# Sensor Cleaning Functions
# ========================

def interpolate_sensor(sensor_df, value_cols, sensor_type, freq='1min'):
    """
    Generic interpolation logic for continuous numeric sensors.

    Args:
        sensor_df (pd.DataFrame): Full sensor dataframe.
        value_cols (list): List of columns to interpolate (e.g., ['x', 'y', 'z']).
        sensor_type (str): Sensor type to filter (e.g., 'accelerometer').
        freq (str): Frequency to resample data to (default: '1min').

    Returns:
        cleaned_all (pd.DataFrame): Interpolated sensor data.
        imputation_log (pd.DataFrame): Count of interpolated values per user and column.
    """
    sensor_df = sensor_df[sensor_df['type'] == sensor_type].copy()
    sensor_df['datetime'] = pd.to_datetime(sensor_df['datetime'])
    sensor_df[value_cols] = sensor_df[value_cols].apply(pd.to_numeric, errors='coerce')

    imputed_logs = []
    cleaned_dfs = []

    # Interpolate per user
    for uid, group in sensor_df.groupby('uid'):
        group = group.set_index('datetime').sort_index()
        resampled = group[value_cols].resample(freq).mean()

        nan_before = resampled.isna().sum()
        resampled = resampled.interpolate(method='linear', limit_direction='both')
        resampled = resampled.ffill().bfill()
        nan_after = resampled.isna().sum()
        imputed_count = (nan_before - nan_after).clip(lower=0)

        for col in value_cols:
            imputed_logs.append({
                'uid': uid,
                'sensor': sensor_type,
                'column': col,
                'imputed': imputed_count[col]
            })

        # Restore metadata
        resampled['uid'] = uid
        resampled['type'] = sensor_type
        resampled = resampled.reset_index()
        cleaned_dfs.append(resampled)

    cleaned_all = pd.concat(cleaned_dfs, ignore_index=True)
    return cleaned_all, pd.DataFrame(imputed_logs)


# ========================
# Per-Sensor Cleaning APIs
# ========================

def clean_accel(sensor_df):
    """
    Clean accelerometer sensor using interpolation on x, y, z axes.
    """
    return interpolate_sensor(sensor_df, ['x', 'y', 'z'], 'accelerometer')


def clean_light(sensor_df):
    """
    Clean light sensor using interpolation on brightness (value).
    """
    return interpolate_sensor(sensor_df, ['value'], 'light')


def clean_wifi(sensor_df):
    """
    Clean Wi-Fi sensor using interpolation on signal strength (level).
    """
    return interpolate_sensor(sensor_df, ['level'], 'wifi')


def clean_location(sensor_df):
    """
    Clean location sensor. Handles missing 'x', 'y', 'z' and converts 'value' (distance) to numeric.
    Does not resample.
    """
    location_df = sensor_df[sensor_df['type'] == 'location'].copy()
    location_df['value'] = pd.to_numeric(location_df.get('value', np.nan), errors='coerce')

    for col in ['x', 'y', 'z']:
        if col in location_df.columns:
            location_df[col] = pd.to_numeric(location_df[col], errors='coerce')
            location_df[col] = location_df[col].interpolate(limit_direction='both')

    location_df = location_df.dropna(subset=['datetime'])

    return location_df


# ========================
# Event-Driven Sensors (No Interpolation)
# ========================

def clean_screen(sensor_df):
    """
    Extract screen sensor events (e.g., ON/OFF) — no interpolation needed.
    """
    screen_df = sensor_df[sensor_df['type'] == 'screen'].copy()
    return screen_df[['uid', 'datetime', 'type', 'value']], pd.DataFrame()


def clean_calls(sensor_df):
    """
    Extract call event records — no interpolation needed.
    """
    calls_df = sensor_df[sensor_df['type'] == 'calls'].copy()
    return calls_df[['uid', 'datetime', 'type', 'sub_type', 'suuid', 'value', 'level', 'sensor_status']], pd.DataFrame()
