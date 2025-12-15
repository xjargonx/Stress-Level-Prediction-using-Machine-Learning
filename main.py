from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# 1. Initialize the App
app = FastAPI(
    title="Stress Level Predictor API",
    description="A Machine Learning API to predict stress levels based on wearable data.",
    version="1.0"
)

# 2. Define the Input Data Format
# This ensures the user sends numbers, not text
class StressInput(BaseModel):
    sleep_hours: float
    activity_minutes: float
    heart_rate: float
    daily_steps: float

# 3. Load the Model
# We check if the file exists to prevent crashing
MODEL_PATH = r"C:\Users\DELL\Study\Semester 5\Machine Learning\Project\Fast API\stress_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    model = None
    print(f"ERROR: Model file '{MODEL_PATH}' not found.")

# 4. Define the Prediction Endpoint
@app.post("/predict")
def predict_stress(data: StressInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found.")
    
    # Prepare features in the exact order the model expects
    features = np.array([[
        data.sleep_hours, 
        data.activity_minutes, 
        data.heart_rate, 
        data.daily_steps
    ]])
    
    # Make Prediction
    prediction = model.predict(features)[0]
    
    # Clip result to 1-10 range and round it
    result = round(np.clip(prediction, 1.0, 10.0), 1)
    
    return {
        "prediction": result,
        "input_data": data
    }

# 5. Define a Root Endpoint (Health Check)
@app.get("/")
def home():
    return {"message": "Stress Predictor API is running. Go to /docs to test it."}