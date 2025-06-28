import os
import mlflow

# Optional: load from .env file
from dotenv import load_dotenv
load_dotenv()

run_id = os.getenv("RUN_ID")

if not run_id:
    raise ValueError("❌ RUN_ID environment variable is not set.")

model_uri = f"runs:/{run_id}/model"
model_name = "CreditRiskLogisticModel"

result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(f"✅ Model registered!\nName: {result.name}\nVersion: {result.version}")
