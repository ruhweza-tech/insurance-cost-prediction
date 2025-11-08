
# Import required libraries 
import pandas as pd 
import numpy as np 
import json 
import joblib
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from pathlib import Path
from pydantic import BaseModel, conint, confloat



# Load .env variables and define the base working directory[two levels up]
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the path to the config and load the configuration variables from json 
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

# Define the path to the trained model
MODEL_PATH = BASE_DIR/config["model_path"]


# Creating the user input data structure with validation
class PredictionInput(BaseModel):
                age:conint(ge = 1, le = 120)
                sex:conint(ge = 0, le = 1)
                bmi:confloat(gt = 0)
                children:conint(ge = 0, le = 10)
                smoker:conint(ge = 0, le = 1)
                northwest:conint(ge = 0, le = 1)
                southeast:conint(ge = 0, le = 1)
                southwest:conint(ge = 0, le = 1)

def create_app(model_path: Path)-> FastAPI:
    """
    Function to create a FastAPI app
        - Load the trained model 
        - Intitialize the app 
        - User input data structure 
        - Create an endpoint
        - sub function to enable users to make predictions 
                - Prepare user data for prediction
                - Make prediction 
                - Return predicted output
        - Return the app
    """
    # Load the trained model 
    model = joblib.load(model_path)

    # Create a fastapi app that will handle all API routes 
    app = FastAPI(
                title = "Insurance cost prediction", 
                description = "Predicting whether insurance cost is low (<2000kr) or high (>2000kr)",
                version = "1.0.0"
                )

    # Creating an endpoint that the users will use to interact with the API
    @app.post(
            path = "/api/prediction",
            summary = "Insurance cost prediction",
            description = "Predicting whether insurance cost is low (<2000kr) or high (>2000kr)",
            tags = ["Prediction"]
             )
    
    def user_prediction(input_data: PredictionInput = Body(...)):
            """
            Sub function to enable users to make predictions 
                - Prepare user input data structure for predictions
                - Make predictions 
                - Return the output
            """

            # Prepare user input data structures for predictions 
            user_input_data = np.array([[
                                        input_data.age,
                                        input_data.sex,
                                        input_data.bmi,
                                        input_data.children,
                                        input_data.smoker,
                                        input_data.northwest,
                                        input_data.southeast,
                                        input_data.southwest
            ]])

            # Make predictions based on user input data 
            predictions = model.predict(user_input_data)[0]
            label = "low insurance cost " if predictions <= 2000 else "high insurance cost"
            return {"Predicted insurance cost in (kr)": round(float(predictions),2),
                    "label": label}

    return app


if __name__== "__main__":
        """
        Single entry point to act as the main execution block
                - Call the function to create the app
                - Run the app locally 
        """
        # Call the fucntion to create the app
        app = create_app(MODEL_PATH)

        # Run the app locally 
        uvicorn.run(app, host = "127.0.0.1", port = 8000)