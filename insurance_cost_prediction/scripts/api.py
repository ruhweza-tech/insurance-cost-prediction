
# Import necessary libraries 
import numpy as np 
import uvicorn
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Define the path for the trained model 
model_path = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\model\model.pkl"

# Load the model 
model = joblib.load(model_path)

# create a fastapi application that handles all API routes 
app = FastAPI(
            title = "Insurance Cost Prediction",
            description = " Predicting wheather the insurance cost is low (< 1000kr) or high (>10000kr)",  
            version = "1.0.0"  
            )

# create a user input data structure 
class PredictionInput(BaseModel):
                        age: int
                        sex:int 
                        bmi:float
                        children:int
                        smoker:int
                        northwest:int
                        southeast:int
                        southwest:int

# create api endpoint where the api will be exposed to users to interact with 
@app.post(
            path = "/api/prediction",
            summary = "Insurance cost prediction",
            description = "Predicting whether the insurance cost is low (< 1000kr) or high (>10000kr)",
            tags = ["Prediction"]
            )
def predict(input_data:PredictionInput):
        
        # Prepare user data for prediction
        data = np.array([[
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
        Prediction = model.predict(data)[0]
        label = "Low insurance cost" if Prediction <= 1000 else "High insurance cost"

        return {"Predicted insurance cost (kr)": round(float(Prediction),2),
                "Label": label}


# create a single entry point 
if __name__ == "__main__":
        uvicorn.run("insurance_cost_prediction.scripts.api:app",host = "127.0.0.1", port = 8000, reload = True)