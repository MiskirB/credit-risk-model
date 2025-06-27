import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import os

def calculate_rfm(df, snapshot_date=None):
    df = df.copy()

    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()

    return rfm


def assign_high_risk_cluster(rfm_df, n_clusters=3, random_state=42):
    # Normalize RFM features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster: typically high Recency + low Frequency + low Monetary
    cluster_profile = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_profile.sort_values(by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]).index[0]

    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    return rfm_df[['CustomerId', 'is_high_risk']]


def main():
    raw_data_path = 'data/raw/data.csv'
    processed_output_path = 'data/processed/labeled_data.csv'

    df = pd.read_csv(raw_data_path)
    rfm = calculate_rfm(df)
    labels = assign_high_risk_cluster(rfm)

    # Merge label into original processed file
    processed_df = pd.read_csv('data/processed/processed_data.csv')
    final_df = pd.merge(processed_df, labels, on='CustomerId', how='left')
    final_df['is_high_risk'] = final_df['is_high_risk'].fillna(0).astype(int)

    os.makedirs('data/processed', exist_ok=True)
    final_df.to_csv(processed_output_path, index=False)
    print(f"✅ Target variable `is_high_risk` created and saved to {processed_output_path}")


if __name__ == "__main__":
    main()
