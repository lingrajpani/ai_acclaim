import pandas as pd

# --- Rolling Feature Generator ---
def add_rolling_features(df, col, windows=[5, 10, 30]):
    """
    Generate rolling mean and standard deviation features for a given column.
    Uses min_periods=1 to avoid all-NaN columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Name of the column to process.
        windows (list): List of window sizes.
        
    Returns:
        pd.DataFrame: Updated DataFrame with rolling features.
    """
    for window in windows:
        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
    return df

# --- Lag Feature Generator ---
def add_lag_features(df, col, lags=[1, 2, 3]):
    """
    Create lag features for a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column to compute lag for.
        lags (list): List of lag periods.
        
    Returns:
        pd.DataFrame: Updated DataFrame with lag features.
    """
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

# --- Change Rate Feature Generator ---
def add_change_rate(df, col):
    """
    Compute the difference (change rate) of a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column name.
        
    Returns:
        pd.DataFrame: DataFrame with an added change rate feature.
    """
    df[f'{col}_change_rate'] = df[col].diff()
    return df

# --- Time-Based Features ---
def add_time_based_features(df, timestamp_col='Timestamp'):
    """
    Extract time-based features (hour, weekday, is weekend) from a timestamp column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        timestamp_col (str): Timestamp column name.
        
    Returns:
        pd.DataFrame: DataFrame with added time-based features (Hour, Weekday, IsWeekend).
    """
    df['Hour'] = df[timestamp_col].dt.hour
    df['Weekday'] = df[timestamp_col].dt.dayofweek
    df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
    return df

# --- Master Feature + Label Builder (Metrics Only) ---
def build_features_with_labels(df, metrics_cols=None, timestamp_col='Timestamp'):
    """
    Build engineered features for system metrics only (no event log logic).
    If metrics_cols is None, use all numeric columns except timestamp/source.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', unit='ms')
    df = df.sort_values(by=timestamp_col)

    # If metrics_cols not provided, use all numeric columns except timestamp/source
    if metrics_cols is None:
        metrics_cols = [col for col in df.select_dtypes(include='number').columns if col not in [timestamp_col, 'Timestamp']]

    for col in metrics_cols:
        if col in df.columns:
            df = add_rolling_features(df, col)
            df = add_lag_features(df, col)
            df = add_change_rate(df, col)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    df = add_time_based_features(df, timestamp_col)
    df = df.dropna(subset=[timestamp_col]).reset_index(drop=True)
    return df

# --- Event Count Rolling Features (deprecated for metrics-only pipeline) ---
def add_event_counts(event_df, timestamp_col='Timestamp', event_type_col='EntryType', window='10min'):
    print("Event-based features are deprecated in the metrics-only pipeline.")
    return pd.DataFrame()

def merge_event_features(metrics_df, event_features_df, on='Timestamp'):
    print("Event-based features are deprecated in the metrics-only pipeline.")
    return metrics_df