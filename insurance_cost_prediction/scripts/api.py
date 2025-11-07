
# Import regquired libraries 
import joblib 
import uvicorn 
import json
import numpy as np 
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, conint, confloat
from pathlib import Path
from typing import Annotated


# Load the .env variables and define the base working directory (two levels up from this file)
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the path to config and load config variables from json
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
        config = json.load(m)

# Define the path to the trained model 
MODEL_PATH = BASE_DIR/config["model_path"]

def create_app(model_path: Path)-> FastAPI:
        """
        Function to create a fast api app
                - Load the trained model
                - Initialize the FastApi
                - User input data structure
                - Create an endpoint
                - Sub function to make predictions
                        - Return output
                - Return app
        """
        # Load the trained model
        model = joblib.load(model_path)

        # Creating a fastapi app that handles all api routes
        app = FastAPI(
                        title = "Insurance cost prediction",
                        description = "Predicting whether the insurance cost is low (<1000kr) or high (>1000kr)",
                        version = "1.0.0"
                     )
        
        # Creating the user input data structure 
        class PredictionInput(BaseModel):
                                age:Annotated[int, conint(ge=0, le=100)]
                                sex:Annotated[int, conint(ge = 0, le = 1)]
                                bmi:Annotated[float, confloat(gt = 0)]
                                children:Annotated[int,conint(ge =0, le = 10)]
                                smoker:Annotated[int,conint(ge = 0, le = 1)]
                                northwest:Annotated[int, conint(ge = 0, le = 1)]
                                southeast:Annotated[int, conint(ge = 0, le = 1)]
                                southwest:Annotated[int, conint(ge = 0, le = 1)]

        # Create an endpoint for users to be able to interact with the api
        @app.post(
                path = "/api/prediction",
                summary = "Insurance Cost Prediction",
                description = "Predicting whether the insurance cost is low (<1000kr) or high (>1000kr)",
                tags = ["Prediction"]
                )
        def predict(input_data:PredictionInput):
                """ 
                Sub function to take user input data and make predictions based on the trained model
                        - Prepare user data for predictions 
                        - Make predictions 
                        - Return the output
                """
                user_data = np.array([[
                                     input_data.age,
                                     input_data.sex,
                                     input_data.bmi,
                                     input_data.children,
                                     input_data.smoker,
                                     input_data.northwest,
                                     input_data.southeast,
                                     input_data.southwest  
                                     ]])
                
                # Make the prediction 
                prediction = model.predict(user_data)[0]
                label = "Low insurance cost" if prediction <= 1000 else "High insurance cost"
                return {"Predicted insurance cost in (kr)": round(float(prediction),2),
                        "label": label}
        return app

if __name__ == "__main__":
        """
        Single entry point as the main execution block
                - Call the fucntion to create the app
                - Run the api locally 
        """
        # Call the function to create the app
        app = create_app(MODEL_PATH)

        # Run the app locally 
        uvicorn.run(app, host = "127.0.0.1", port = 8000)

