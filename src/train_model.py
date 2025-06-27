import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- Data Splitter ---
def split_data(df, target='failure_coming', test_size=0.2, random_state=42):
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    if target in non_numeric_columns:
        non_numeric_columns.remove(target)
    X = df.drop(columns=[col for col in non_numeric_columns if col != target])
    if target in X.columns:
        X = X.drop(columns=[target])  # Ensure target is not in features
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# --- Model Trainer ---
def train_xgboost_classifier(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

# --- Model Evaluator ---
def evaluate_model(model, X_test, y_test, show_plots=True):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC Score:", roc_auc)
    if show_plots:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()
        xgb.plot_importance(model, max_num_features=10, importance_type='gain')
        plt.title("XGBoost Feature Importance")
        plt.show()
    return roc_auc

# --- Model Saver ---
def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to: {path}")

# --- Model Loader ---
def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

# --- Model Retrainer ---
def retrain_model(model, X_train, y_train):
    model.fit(X_train, y_train, xgb_model=model.get_booster())
    return model
