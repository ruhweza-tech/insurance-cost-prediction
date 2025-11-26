
# Import required libraries 
import uvicorn 
import numpy as np 
from fastapi import FastAPI, Body
from pathlib import Path 
import joblib
import os
import json 
from pydantic import BaseModel, conint, confloat
from dotenv import load_dotenv


# Load the .env and define the base working directory 
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent

# Identify the path to config and load it's variables 
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Identify the path to the trained model
MODEL_PATH = BASE_DIR/config["model_path"]

# Create the user input data structure with validation constraints 
class PredictionInput(BaseModel):
                age:conint(ge = 1, le = 120)
                sex:conint(ge = 0, le = 1)
                bmi:confloat(gt = 0)
                children:conint(ge = 0, le = 1)
                smoker:conint(ge = 0, le = 1)
                northwest:conint(ge = 0, le = 1)
                southeast:conint(ge = 0, le = 1)
                southwest:conint(ge = 0, le = 1)

#================== FastAPI FACTORY ==================#

def create_app(model_path: Path)-> FastAPI:
        """
        Function to create a fastAPI app
                - Load the trained model 
                - Initialize the fastapi
                - Create an endpoint 
                - Sub function to make predictions 
                        - Prepare user input data for predictions 
                        - Make predictions 
                        - Return the output
                - Return the app
        """
        # Load the trained model 
        model = joblib.load(model_path)

        # Create the fast api app that will handle all API routes
        app = FastAPI(
                title = "Insurance cost prediction",
                description = "Predicting whether the insurance cost is low (<=1000kr) or high (>=1000kr)",
                version = "1.0.0"
                )
        
        # Create an endpoint the users can later use to interact with the app i.e. were they will post their input data
        @app.post(
                path = "/api/prediction",
                summary = "Insurance cost prediction",
                description = "Predicting whether the insurance cost is low (<=1000kr) or high (>=1000kr)",
                tags = ["Prediction"]
                 )
        
        def predictions(input_data: PredictionInput = Body(...)):
                """
                Sub fucntion to take user input to make predictions 
                        - Prepare user input data 
                        - Make predictions 
                """
                # Prepare user input data for predictions by turning it to 2D array
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
                
                # Take user input data and make predictions 
                pred = model.predict(user_input_data)
                label = "Low insurance cost" if pred <= 1000 else "high insurance cost"

                # Return the output 
                return  {"The predicted insurance cost in kr is": round(float(pred),2),
                         "lable": label}
        
        # Return the app
        return app
    
# Calling the function to create the app
app = create_app(MODEL_PATH)

if __name__ =="__main__":
        """
        Single entry point acting as the main execution block
                - Run the fast api locally 
        """
        # Use default port 
        port = int(os.environ.get("PORT", 8000))

        # Run the fastapi locally 
        uvicorn.run(app, host = "0.0.0.0", port = port)