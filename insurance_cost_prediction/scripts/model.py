# Import required libraries 
import numpy as np 
import pandas as pd
import json 
import joblib
import os
import xgboost as xgb
from pathlib import Path 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from dotenv import load_dotenv

# Load the .env variables and define the base working directory 
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Identify the path to config and also load it's variables from json 
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Identify the path to the processed dataset and the path to the folder that will store the trained model
CLEAN_PATH = BASE_DIR/config["clean_path"]
MODEL_FOLDER = BASE_DIR/config["model_folder"]

# Check if the model folder exists, if missing create one automatically 
MODEL_FOLDER.mkdir(parents = True, exist_ok = True)

# Create a full path in the model folder to store the trained model
MODEL_PATH = MODEL_FOLDER/"model.pkl"

def train_model(clean_path_data: Path):
    """
    Fuunction to train the model using the processed dataset
        - Load the clean dataset 
        - Extract the independent and dependent values 
        - Split the dataset into train and test sets 
        - Build and train the model 
        - Inference
        - Evaluation 
        - Return results
    """
    # Load the clean dataset 
    processed_data = pd.read_csv(clean_path_data)

    # Extract the independent and dependent dataset
    x = processed_data.iloc[:,:-1].values 
    y = processed_data.iloc[:,-1].values

    # Split the dataset into train and test sets 
    x_train, x_test, y_train, y_test = train_test_split(x, y, **config["split_params"])

    # Build and train the model 
    model = xgb.XGBRegressor(**config["xgb_params"])
    model.fit(x_train, y_train)

    # Inference i.e. make predictions on the current dataset
    y_pred = model.predict(x_test)

    # Build a dataframe to compare the predicted values vs the true values 
    comparison_df = pd.DataFrame({
                                "Predicted values": np.round(y_pred,2),
                                "True values":np.round(y_test,2),
                                "Errors":np.round(y_pred - y_test)

                                })
    
    print("\n=================================================================")
    print(comparison_df.sample(5))
    

    # Evaluation
    ## Caculate the R-Squared 
    r2 = r2_score(y_test, y_pred)

    ## Calculate the Adjusted R-Squared 
    k = x_test.shape[1]     # Number of columns 
    n = x_test.shape[0]     # Number of rows

    adj_r2 = 1 - (1 - r2) * (n - 1)/(n - k - 1)

    ## Calculate the K-Fold Cross Validation 
    avg_r2 = cross_val_score(
                            estimator = model,
                            X = x, 
                            y = y, 
                            scoring = "r2",
                            cv = 10
                            )
    print("\n=================================================================")
    print(f"The average R-Squared (10-fold): {avg_r2.mean():.3f}")
    print(f"The standard deviation is : {avg_r2.std():.3f}")
    

    # Return the results 
    return model, r2, adj_r2, avg_r2

if __name__ == "__main__":
    """
    Single entry point to act as the main execution block for this script 
        - Call the function to train the model 
        - Save the model 
        - Write the performance metrics to json 
        - Create a path in the Model folder to store the performance metrics 
        - Store the performance metrics to json file 
        - Display messages 
    """

    # Call the function to train the model 
    model, r2, adj_r2, avg_r2 = train_model(CLEAN_PATH)

    # Save the trained model to the path 
    joblib.dump(model, MODEL_PATH)

    # Write model performance metrics to json file
    metrics = {
                "R-Squared": r2,
                "Adjusted R-Squared": adj_r2,
                "Average R-Squared(10-fold)": avg_r2.mean(),
                "Standard Deviation": avg_r2.std()
                }
    
    # Create a full path in the model folder to store the performance metrics 
    metrics_path = MODEL_FOLDER/"performance_metrics.json"

    # Save the model performance metrics to the path 
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent = 4)

    # Display messages and results 

    print("\n=================================================================")
    print(f"The Trained model is stored at : {MODEL_PATH}")
    print(" ")
    print(f"The Model performance metrics are stored at : {metrics_path}")
    print(" ")
    print(f"The Environment is: {os.getenv("ENVIRONMENT","local")}")
    print(f"The Author is : {os.getenv("AUTHOR","Mugisha")}")
    print("=================================================================\n")








