import mlflow.artifacts

model_uri = "models:/CreditRiskLogisticModel/1"
local_path = "model"

mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=local_path)

print(f"✅ Model downloaded to: {local_path}")
