import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_TRAIN_PATH = "data/processed/X_train.csv"
Y_TRAIN_PATH = "data/processed/y_train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")

def train_model():
    # Create output directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load separate files
    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH).values.ravel()  # Flatten y

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
