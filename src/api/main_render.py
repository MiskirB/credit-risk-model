from fastapi import FastAPI
from src.api.pydantic_models import CustomerInput, RiskPrediction
import mlflow.pyfunc


app = FastAPI()

model = mlflow.pyfunc.load_model("model")
@app.get("/")
def root():
    return {"message": "✅ Welcome to the Credit Risk Prediction API!"}

@app.post("/predict", response_model=RiskPrediction)
def predict_risk(customer: CustomerInput):
    input_df = customer.to_dataframe()
    risk_score = model.predict(input_df)[0]
    return RiskPrediction(risk_probability=float(risk_score))
