
# Import the required libraries
import numpy as np 
import pandas as pd 
import json
import joblib
from dotenv import load_dotenv
from pathlib import Path 
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score


# Load .env variables and define the base working directory[2 levels up]
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the path to config and load it's variables from json
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

# Define the path to the clean dataset and the folder to store the trained model
CLEAN_DATASET = BASE_DIR/config["clean_dataset_path"]
MODEL_FOLDER = BASE_DIR/config["model_path"]

# Checking if the model folder exists, if missing , create one automatically 
MODEL_FOLDER.mkdir(parents = True, exist_ok = True)

# Create a full path in the model folder to store the trained model
TRAINED_MODEL = MODEL_FOLDER/"model.pkl"

def train_model(clean_dataset: Path):
    """
    Function to train the model
        - Load the clean dataset
        - Extract the independent and dependent variables 
        - Split the dataset into train and test sets 
        - Build and train the model
        - Inference
        - Evaluation
    """

    # Load the clean dataset
    clean_data = pd.read_csv(clean_dataset)

    # Extract the independent and dependent variables 
    x = clean_data.iloc[:,:-1].values
    y = clean_data.iloc[:,-1].values

    # Split the clean data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, **config["split_params"])

    # Build and train the model 
    model = xgb.XGBRegressor(**config["xgb_params"])
    model.fit(x_train, y_train)

    # Make predictions on the cleaned dataset
    y_pred = model.predict(x_test)

    # Creating a DataFrame to compare the predicted values vs the true values 
    comparison_df = pd.DataFrame({
                                "Predicted values": np.round(y_pred,),
                                "True values": np.round(y_test,2),
                                "Error": np.round(y_pred - y_test)
                                })
    print("\n==================================================================")
    print(comparison_df.sample(5))
    print("==================================================================\n")
    
    ## Evaluation R-Squared, Adjusted R-Sqaured, K-Fold cross validation

    # Calculate the R-Squared
    r2 = r2_score(y_test, y_pred)

    # Calculate the Adjusted R-Squared
    k = x_test.shape[1] # Determines the number of columns 
    n = x_test.shape[0] # Determines the number of rows

    adj_r2 = 1 - (1 - r2) * (n - 1)/(n - k - 1)

    # Determine the K-FOLD cross validation
    avg_r2 = cross_val_score(
                        estimator = model,
                        X = x,
                        y = y, 
                        scoring = "r2",
                        cv = 10
                        )
    print(" ")
    print(f"The average R-Sqaured (10-fold): {avg_r2.mean():.3f}")
    print(f"Standard Devaition: {avg_r2.std():.3f}")

    # Return the trained model and evaluation metrics
    return model, r2, adj_r2, avg_r2

if __name__=="__main__":
    """
    Single point entry to act as the main execution block for this script
        - Call the function to train the model 
        - Save the model 
        - write the performance metrics to json
        - Create a path to store the performance metrics
        - Put them into json file and store the performance metrics
        - Display the output
    """
    # Call the fucntion to train the model on the clean dataset
    model, r2, adj_r2, avg_r2 = train_model(CLEAN_DATASET)
    
    # Save the model to the new path 
    joblib.dump(model, TRAINED_MODEL)

    # Write model performance metrics to json 
    metrics = {
                "R-Sqaured":r2,
                "Adjusted R-Squared":adj_r2,
                "Average R-Squared(10-fold)":avg_r2.mean(),
                "Standard Devaition": avg_r2.std()
                }
    # Create a full path in the model folder to store the performance metrics
    metrics_path = MODEL_FOLDER/"performance_metrics.json"

    # Write the performance metrics to json 
    with open(metrics_path, "w") as m:
        json.dump(metrics, m, indent = 4)

    # Displaying the output 
    print("\n==================================================================")
    print(f"The trained model is saved at: {TRAINED_MODEL}")
    print(f"The model performance metrics are saved at: {metrics_path}")
    print("==================================================================\n")
    