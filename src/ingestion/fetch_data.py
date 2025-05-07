import pandas as pd
import os

def download_data():
    url = "https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv"

    output_path = os.path.join("data", "raw", "churn_data.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(url)
    df.to_csv(output_path, index=False)
    print(f"Data downloaded to {output_path}")

if __name__ == "__main__":
    download_data()
