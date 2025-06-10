import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

# Load the model
model_path = os.path.join("models", "model.joblib")
try:
    model = joblib.load(model_path)
    logging.info(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    logging.error(f"Model file not found at {model_path}. Make sure to train the model first.")
    model = None

# Define the input data structure using Pydantic
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Wine Quality Prediction API!"}

@app.post("/predict")
async def predict(request: Request, features: WineFeatures):
    if model is None:
        return {"error": "Model is not loaded. Please train the model first."}, 500

    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Interpret the prediction
        quality = "Good" if prediction[0] == 1 else "Bad"
        
        return {
            "prediction": quality,
            "probability_good": f"{probability[0][1]:.4f}"
        }
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": "An error occurred during prediction."}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 