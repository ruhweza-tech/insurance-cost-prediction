
# Import the required libraries 
import pandas as pd 
import json 
import numpy as np 
import joblib
import xgboost as xgb 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from pathlib import Path 
from dotenv import load_dotenv

# Load the .env variables and define the base working directory 
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the path to config and load the configuration variables from the json file 
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Define the path where the clean dataset is stored and the folder to store the trained model
CLEAN_PATH = BASE_DIR/config["clean_dataset_path"]
MODEL_PATH = BASE_DIR/config["model_path"]

# Check if the folder to store the trained model exists , if missing , automatically create one
MODEL_PATH.mkdir(parents = True, exist_ok = True)

# Create a full path in the model folder to store the trained model
SAVE_MODEL = MODEL_PATH/"model.pkl"

def train_model(CLEAN_PATH: Path):
    """ 
    Function to train the model using xgboost
        - Load the cleaned dataset
        - Extract the independent and dependent variables 
        - Split the dataset into train and test sets
        - Build and train the model
        - Inference
        - Evaluation
    """

    # Load the cleaned dataset
    clean_dataset = pd.read_csv(CLEAN_PATH)

    # Extract the independent and dependent variables 
    x = clean_dataset.iloc[:,:-1].values
    y = clean_dataset.iloc[:,-1].values

    # Split the cleaned dataset into train and test sets 
    split_params = config["split_params"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, **split_params)

    # Build and train the model 
    model = xgb.XGBRegressor(**config["xgb_params"])
    model.fit(x_train, y_train)

    # Make predictions and compare the predictions side to side with the true values 
    y_pred = model.predict(x_test)

    # Create a dataframe to compare the predicted values to the true values
    comparison_df = pd.DataFrame({
                                "Predicted values":np.round(y_pred, 2),
                                "True values": np.round(y_test,2),
                                "Error":np.round(y_test - y_pred, 2)
                                })
    print(f"\nPredicted values vs True values vs Error \n")
    print(comparison_df.sample(10))

    # Evaluation 
    ## R-Squared
    r2 = r2_score(y_test, y_pred)

    ## Adjusted R-Squared
    k = x_test.shape[1] # number of columns 
    n = x_test.shape[0] # number of rows
    adj_r2 = 1 - (1 - r2) * (n - k)/(n - k - 1)

    ## K-Fold cross validation
    avg_r2 = cross_val_score(
                            estimator = model,
                            X = x,
                            y = y,
                            scoring = "r2",
                            cv = 10
                            )
    print("  ")
    print(f"The average R-Squared (10-fold):{avg_r2.mean():.3f}")
    print(f"The standard deviation: {avg_r2.std():.3f}")
    
    # Return the trained model and the model performance metrics 
    return model, r2, adj_r2, avg_r2

if __name__ == "__main__":
    """ 
    Single entry point for all the execution of the script
        - Load the function to train the model
        - Save the model to the full path in the model folder 
        - Save model performance metrics as json
        - Create a new path in the model folder to store the performance metrics 
        - Write the model performance metrics to json file
        - Display the output 
    """

    # Load the function to train the model 
    model, r2, adj_r2, avg_r2 = train_model(CLEAN_PATH)

    # Save the model to full path created earlier on in the model folder 
    joblib.dump(model, SAVE_MODEL)

    # Save model performance metrics as json
    metrics = {
                "R-Squared": r2,
                "Adjusted R-Squared": adj_r2,
                "Average R-Squared (10-fold)": avg_r2.mean(),
                "Standard Deviation": avg_r2.std()
                }
    
    # Create a full path in the model folder to store the performance metrics 
    metrics_path = MODEL_PATH/"performance_metrics.json"

    # Write the model performance metrics to json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent = 4)

    # Display the output 
    print("\n=======================================================================================")
    print(f"The trained model is saved at: {SAVE_MODEL}")
    print(f"The model performance metrics are saved at: {metrics_path}")
    print("=======================================================================================\n")

    print(BASE_DIR)
