import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

# 1. Custom transformer to extract time-based features
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors='coerce')
        df['transaction_hour'] = df[self.time_col].dt.hour
        df['transaction_day'] = df[self.time_col].dt.day
        df['transaction_month'] = df[self.time_col].dt.month
        df['transaction_year'] = df[self.time_col].dt.year
        return df.drop(columns=[self.time_col])


# 2. Main function to process data
def build_feature_pipeline(df):
    df = df.copy()

    # Remove unused columns
    drop_cols = ['TransactionId', 'BatchId', 'SubscriptionId', 'CountryCode']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Feature: total number of transactions per customer
    df['TransactionCount'] = df.groupby('CustomerId')['CustomerId'].transform('count')

    # Aggregate Amount, Value per customer
    df['TotalAmount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    df['AvgAmount'] = df.groupby('CustomerId')['Amount'].transform('mean')
    df['StdAmount'] = df.groupby('CustomerId')['Amount'].transform('std')
    df['TotalValue'] = df.groupby('CustomerId')['Value'].transform('sum')

    # Extract time features
    df = TimeFeatureExtractor().fit_transform(df)

    # Define columns for pipeline
    num_cols = ['Amount', 'Value', 'TransactionCount', 'TotalAmount', 'AvgAmount', 'StdAmount', 'TotalValue',
                'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']
    cat_cols = ['ChannelId', 'ProductCategory', 'PricingStrategy']

    # Fill NaNs for numerical columns
    df[num_cols] = df[num_cols].fillna(0)

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    X = full_pipeline.fit_transform(df)

    return X, df  # Return processed NumPy array and processed DataFrame


# Optional: Script mode (so you can run `python src/data_processing.py`)
if __name__ == "__main__":
    import os

    # Load data
    input_path = 'data/raw/data.csv'
    output_path = 'data/processed/processed_data.csv'
    df = pd.read_csv(input_path)

    # Run pipeline
    X, df_processed = build_feature_pipeline(df)

    # Save result
    os.makedirs('data/processed', exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"✅ Feature engineering complete. Saved to {output_path}")
