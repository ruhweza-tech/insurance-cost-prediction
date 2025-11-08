
# Import required libraries 
import joblib 
import json 
import uvicorn
import numpy as np 
from pathlib import Path 
from fastapi import FastAPI, Body
from pydantic import BaseModel, conint, confloat


# Load the .env variables and define the base working directory[2 levels up]
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the path to the config and load the configuration variables from json 
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

# Define the path to the trained model 
MODEL_PATH = BASE_DIR/config["model_path"]

# Create the user input data structure with validation constraints
class PredictionInput(BaseModel):
                age:conint(ge = 1, le = 120)
                sex:conint(ge = 0, le = 1)
                bmi:confloat(gt = 0)
                children:conint(ge = 0, le = 10)
                smoker:conint(ge = 0, le = 1)
                northwest:conint(ge = 0, le = 1)
                southeast:conint(ge = 0, le = 1)
                southwest:conint(ge = 0, le = 1)

def create_app(model_path:Path)-> FastAPI:
        """
        Fucntion to create fastapi app
                - Load the trained model 
                - Initialize the fastapi
                - Create an endpoint
                - Sub fucntion to make predictions
                        - Prepare user input data for predictions 
                        - Make predictions 
                        Return the output
                - Return the app
        """
        # Load the trained model 
        #model = joblib.load(model_path)

        # Creating a fastapi app that handles all API routes 
        app = FastAPI(
                title = "Insurance Cost Prediction",
                description = "Predicting whether insurance cost is low (<2000kr) or high (>2000kr)",
                version = "1.0.0"
                     )
        
        # Creating an endpoint that users can use to interact with the API
        @app.post(
                path = "/api/prediction",
                summary = "Insurance Cost Prediction",
                description = "Predicting whether insurance cost is low (<2000kr) or high (>2000kr)",
                tags = ["Prediction"]
                )
        
        def prediction(input_data:PredictionInput = Body(...)):
                """
                Sub fucntion to take user input data and make predictions 
                        - Prepare user input data for prediction 
                        - Make predictions 
                """
                 # Load the trained model each time
                model = joblib.load(model_path)

                # Prepare user input data from the user data structure 
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

                # Make predictions based on the user input data 
                prediction = model.predict(user_input_data)
                label = "Low insurance cost" if prediction <= 2000 else "High insurance cost"

                # Return the predicted output 
                return {"The predicted insurance cost is (kr)": round(float(prediction),2),
                        "label": label}
        
        # Return the app
        return app


if __name__ == "__main__":
        """
        Single entry point acting as the main execution block 
                - Call the function to create the app
                - Run the fast api locally 
        """
        # Call the fucntion to create the fastapi app
        app = create_app(MODEL_PATH)

        # Run the fastapi locally 
        uvicorn.run(app, host = "127.0.0.1", port = 8000)

        