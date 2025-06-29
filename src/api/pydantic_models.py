from pydantic import BaseModel
import pandas as pd


class CustomerInput(BaseModel):
    Amount: float
    Value: int
    PricingStrategy: int
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    # Add any other relevant features

    
    
    def to_dataframe(self):
        return pd.DataFrame([self.dict()])
    


class RiskPrediction(BaseModel):
    risk_probability: float
