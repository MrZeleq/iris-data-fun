import os

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

# Load the pre-trained model from the specified path
model = load(os.path.join('..', 'model', 'lr_model.joblib'))

# Define a Pydantic model for Iris data
class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create a FastAPI instance
app = FastAPI()

# Define a POST endpoint for prediction
@app.post('/predict')
async def predict_species(iris: Iris):
    # Prepare data for prediction
    data = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    
    # Make a prediction
    prediction = model.predict(data)
    
    # Return the prediction result
    return {'species': prediction[0]}