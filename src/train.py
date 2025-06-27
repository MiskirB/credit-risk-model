import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import mlflow
import mlflow.sklearn

# Ensure src/ is on the path and import feature pipeline
sys.path.append(str(Path(__file__).resolve().parents[0]))
from data_processing import build_feature_pipeline


def load_data():
    # Load raw data
    raw_path = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'data.csv'
    raw_df = pd.read_csv(raw_path)

    # Load labels and merge FIRST
    label_path = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'labeled_data.csv'
    labels = pd.read_csv(label_path)[['CustomerId', 'is_high_risk']]

    df = raw_df.merge(labels, on='CustomerId', how='inner')
    y = df['is_high_risk']

    # Apply feature engineering to filtered df
    X_processed, _ = build_feature_pipeline(df)

    return train_test_split(X_processed, y, test_size=0.2, random_state=42)



def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        mlflow.log_param("model_name", model_name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)

        print(f"\n✅ {model_name} Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print(f"📦 Model logged to MLflow as '{model_name}'\n")


def main():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Credit Risk Modeling")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        train_and_log_model(model, name, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
