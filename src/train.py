import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Add src/ to path for imports
sys.path.append(str(Path(__file__).resolve().parents[0]))
from data_processing import build_feature_pipeline


def load_data():
    raw_path = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'data.csv'
    label_path = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'labeled_data.csv'

    raw_df = pd.read_csv(raw_path)
    labels = pd.read_csv(label_path)[['CustomerId', 'is_high_risk']]

    df = raw_df.merge(labels, on='CustomerId', how='inner')
    y = df['is_high_risk']

    # ✅ Build features — ensure result is DataFrame
    X_processed, _ = build_feature_pipeline(df)

    # ✅ If output is NumPy array, force convert to DataFrame
    if not isinstance(X_processed, pd.DataFrame):
        X_processed = pd.DataFrame(X_processed)

    # ✅ Split and reassign column names if needed
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # ✅ Ensure column names are retained
    if hasattr(X_processed, 'columns'):
        X_train.columns = X_processed.columns
        X_test.columns = X_processed.columns

    return X_train, X_test, y_train, y_test



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

        # Log model with input example and signature
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            input_example=input_example,
            signature=signature
        )

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        roc_path = f"roc_curve_{model_name}.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        pr_path = f"pr_curve_{model_name}.png"
        plt.savefig(pr_path)
        mlflow.log_artifact(pr_path)
        plt.close()

        # Confusion Matrix
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        disp.ax_.set_title("Confusion Matrix")
        cm_path = f"conf_matrix_{model_name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Threshold tuning
        thresholds = [0.5, 0.3, 0.2, 0.1]
        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            mlflow.log_metric(f"recall_at_{t}", recall_score(y_test, preds))
            mlflow.log_metric(f"precision_at_{t}", precision_score(y_test, preds))

        print(f"\n✅ {model_name} Results (Threshold=0.5):")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print(f"📦 Model logged to MLflow as '{model_name}'\n")

        # Clean up saved plot images
        for file in [roc_path, pr_path, cm_path]:
            if os.path.exists(file):
                os.remove(file)


def main():
    try:
        X_train, X_test, y_train, y_test = load_data()
        print("✅ Data loaded successfully")

        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("Credit Risk Modeling")

        models = {
            "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        }

        for name, model in models.items():
            print(f"🔍 Training {name}")
            train_and_log_model(model, name, X_train, X_test, y_train, y_test)

    except Exception as e:
        print("❌ Error during training:", e)


if __name__ == "__main__":
    main()
