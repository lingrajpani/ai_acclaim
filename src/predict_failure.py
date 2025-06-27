import os
import pandas as pd
import numpy as np
import pickle

def load_model(model_path):
    """Load a trained XGBoost model from a pickle file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def prepare_features(new_data_path, train_columns, target="failure_coming"):
    """Prepare new data for prediction: align columns, drop non-numeric, fill missing."""
    new_df = pd.read_json(new_data_path)
    if "Timestamp" in new_df.columns:
        new_df["Timestamp"] = pd.to_datetime(new_df["Timestamp"], errors="coerce")
        new_df = new_df.sort_values("Timestamp")
    non_numeric_columns = new_df.select_dtypes(exclude=["number"]).columns.tolist()
    if target in non_numeric_columns:
        non_numeric_columns.remove(target)
    processed_new_df = new_df.drop(columns=non_numeric_columns + [target], errors="ignore")
    processed_new_df_aligned = processed_new_df.reindex(columns=train_columns).fillna(0)
    return processed_new_df_aligned, new_df


def predict_failures(model_path, new_data_path, train_columns, target="failure_coming"):
    """Load model, prepare features, make predictions, and return results DataFrame."""
    model = load_model(model_path)
    processed_new_df_aligned, new_df = prepare_features(new_data_path, train_columns, target)
    predictions = model.predict(processed_new_df_aligned)
    probabilities = model.predict_proba(processed_new_df_aligned)[:, 1]
    results_df = pd.DataFrame({
        "Predicted_Failure": predictions,
        "Failure_Probability": probabilities
    }, index=processed_new_df_aligned.index)
    # Merge with timestamps for context
    if "Timestamp" in new_df.columns:
        results_with_timestamps = new_df[["Timestamp"]].copy()
        results_with_timestamps = results_with_timestamps.merge(results_df, left_index=True, right_index=True)
    else:
        results_with_timestamps = results_df
    return results_with_timestamps

if __name__ == "__main__":
    # Example usage for CLI/testing
    import sys
    if len(sys.argv) < 4:
        print("Usage: python predict_failure.py <model_path> <new_data_path> <train_columns_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    new_data_path = sys.argv[2]
    train_columns_path = sys.argv[3]
    train_columns = pd.read_json(train_columns_path).columns.tolist()
    results = predict_failures(model_path, new_data_path, train_columns)
    print(results.head())
