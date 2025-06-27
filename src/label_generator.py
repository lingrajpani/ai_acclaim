import pandas as pd

# --- Synthetic Labeling Based on Anomalies for System Metrics Only ---
def generate_synthetic_labels(df, metric_cols=None, threshold_std=2.5, window_minutes=10, timestamp_col='Timestamp'):
    """
    Generate failure labels based only on system metrics anomalies.
    Args:
        df (pd.DataFrame): DataFrame with system metrics.
        metric_cols (list): List of metric columns to use for anomaly detection. If None, use all numeric columns except Timestamp.
        threshold_std (float): Number of std deviations above mean to consider anomaly.
        window_minutes (int): Lead time in minutes to propagate label backward.
        timestamp_col (str): Name of the timestamp column.
    Returns:
        pd.DataFrame: DataFrame with 'failure_coming' label column.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', unit='ms')
    df = df.sort_values(timestamp_col)
    df['failure_coming'] = 0

    # If metric_cols not provided, use all numeric columns except timestamp/source
    if metric_cols is None:
        metric_cols = [col for col in df.select_dtypes(include='number').columns if col not in [timestamp_col, 'Timestamp']]

    for col in metric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            threshold = mean_val + threshold_std * std_val
            anomaly_mask = df[col] > threshold
            df.loc[anomaly_mask, 'failure_coming'] = 1

    # Propagate label backward to simulate lead time
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df['failure_coming_rolled'] = 0
    failure_indices = df.index[df['failure_coming'] == 1]
    for idx in failure_indices:
        start_time = df.loc[idx, timestamp_col] - pd.Timedelta(minutes=window_minutes)
        df.loc[(df[timestamp_col] >= start_time) & (df.index < idx), 'failure_coming_rolled'] = 1
    df['failure_coming'] = df['failure_coming_rolled']
    df.drop(columns=['failure_coming_rolled'], inplace=True)
    return df

# --- Event-Based Labeling (deprecated for metrics-only pipeline) ---
def label_from_eventid(df, fail_event_ids=[6008, 1001], lead_minutes=10, timestamp_col='Timestamp'):
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df['failure_coming'] = 0

    if 'EventID' not in df.columns:
        print("EventID column not found. Skipping event-based labeling.")
        return df

    failure_times = df[df['EventID'].isin(fail_event_ids)][timestamp_col]

    for fail_time in failure_times:
        lead_start = fail_time - pd.Timedelta(minutes=lead_minutes)
        df.loc[(df[timestamp_col] >= lead_start) & (df[timestamp_col] < fail_time), 'failure_coming'] = 1

    return df
