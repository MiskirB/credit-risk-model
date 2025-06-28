import mlflow

model_uri = "models:/CreditRiskLogisticModel/1"
local_path = "model"

local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=local_path)
print(f"✅ Model downloaded to: {local_dir}")
