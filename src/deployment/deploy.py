from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Load the trained model
MODEL_PATH = "models/churn_model.pkl"
model = joblib.load(MODEL_PATH)

# Define input schema
class ChurnRequest(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Churn prediction API is running!"}

@app.post("/predict")
def predict(data: ChurnRequest):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {
        "prediction": int(prediction),
        "message": "Churn" if prediction == 1 else "No Churn"
    }

# ✅ THIS PART STARTS THE SERVER
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
