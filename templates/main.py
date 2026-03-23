
# templates/main.py - Example FastAPI application for ML model serving

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="ML Model Serving API", version="1.0.0")

# Load pre-trained model (example: a scikit-learn model)
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl not found. Please train and save a model first.")
    model = None # Handle case where model is not found

# Define request body for prediction
class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predicts output based on input features.
    """
    if model is None:
        return {"error": "Model not loaded. Cannot make predictions."}, 500
    
    try:
        input_data = np.array(request.features).reshape(1, -1)
        prediction = model.predict(input_data).tolist()
        return {"prediction": prediction}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}, 400

# Example of how to train and save a dummy model (for demonstration)
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    # Generate dummy data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    
    # Train a simple model
    dummy_model = LogisticRegression()
    dummy_model.fit(X, y)
    
    # Save the model
    joblib.dump(dummy_model, "model.pkl")
    print("Dummy model trained and saved as model.pkl")
    print("To run this FastAPI app, use: uvicorn main:app --host 0.0.0.0 --port 8000")

# This file provides a basic FastAPI application for serving a machine learning model.
# It demonstrates how to create API endpoints for health checks and predictions.
# The model is loaded using joblib, a common library for serializing Python objects.
# Pydantic BaseModel is used for request body validation, ensuring data integrity.
# Error handling is included for cases where the model is not found or prediction fails.
# This template is essential for MLOps, enabling seamless deployment of ML models.
# It promotes a microservice architecture for ML applications.
# Further development could involve more complex input/output schemas, authentication,
# and integration with monitoring tools.
# This is a practical example of putting ML models into production.
# The comments explain the different parts of the FastAPI application.
# It's a valuable resource for MLOps engineers and data scientists.
# Enjoy deploying your ML models as robust APIs!
