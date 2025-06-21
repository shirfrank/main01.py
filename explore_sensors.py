import pandas as pd

def explore_sensors(sensor_df, session_name):
    """
    Prints a summary of the types and number of sensor samples in a session.

    Args:
        sensor_df (pd.DataFrame): The full dataframe containing sensor data.
                                  Must include a 'type' column indicating the sensor type.
        session_name (str): The name or label of the session (e.g., "Session A") for display purposes.

    Example output:
        --- Exploring sensors in Session A ---
        Found sensor types: ['accelerometer' 'light' 'screen' 'calls']
        Sensor 'accelerometer': 3855 samples
        Sensor 'light': 2982 samples
        ...
    """
    print(f"\n--- Exploring sensors in {session_name} ---")

    # Extract all unique sensor types
    sensor_types = sensor_df['type'].unique()
    print(f"Found sensor types: {sensor_types}\n")

    # Count the number of entries per sensor type
    for sensor_type in sensor_types:
        count = sensor_df[sensor_df['type'] == sensor_type].shape[0]
        print(f"Sensor '{sensor_type}': {count} samples")
