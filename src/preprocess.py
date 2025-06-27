import os
import pandas as pd
import traceback

def load_csv_with_datetime(path, timestamp_col, rename_to='Timestamp'):
    """
    Loads a CSV file, trims the column names, and converts the specified timestamp column.
    
    """
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if timestamp_col in df.columns:
            df[rename_to] = pd.to_datetime(df[timestamp_col], errors='coerce')
        else:
            print(f"Timestamp column '{timestamp_col}' not found in {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"Failed loading {path}: {e}")
        return pd.DataFrame()

def process_system_metrics(system_metric_path):
    """
    Loads only the system metrics CSV, sorts by Timestamp, and returns the processed DataFrame.
    
    Args:
        system_metric_path (str): Path to the system metrics CSV file.
    
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    try:
        sys_df = load_csv_with_datetime(system_metric_path, 'Timestamp')
        sys_df['source'] = 'system_metrics'
        if 'Timestamp' in sys_df.columns:
            sys_df = sys_df.sort_values('Timestamp')
        print(f"Shape: {sys_df.shape}")
        print(sys_df.head())
        return sys_df
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    # Allow passing file paths via command-line arguments if running directly from the terminal.
    import argparse
    parser = argparse.ArgumentParser(
        description="Process only the system metrics CSV file and return a DataFrame."
    )
    parser.add_argument("--system_metric", type=str, required=True, 
                        help="Path to the system metric CSV file.")
    args = parser.parse_args()
    process_system_metrics(args.system_metric)
