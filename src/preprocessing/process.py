import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess():
    input_path = os.path.join("data", "raw", "churn_data.csv")
    df = pd.read_csv(input_path)

    # Drop customerID (not useful)
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Binary encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical features
    cat_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Split into train and test
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save to disk
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("Preprocessing completed and files saved in data/processed/")

if __name__ == "__main__":
    preprocess()
