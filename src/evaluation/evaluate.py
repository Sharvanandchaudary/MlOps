import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Paths
MODEL_PATH = "models/churn_model.pkl"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

def evaluate_model():
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Load test data
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()  # Flatten y

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("📊 Evaluation Metrics:")
    print(f"Accuracy     : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision    : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall       : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score     : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == "__main__":
    evaluate_model()
