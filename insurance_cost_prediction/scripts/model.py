#=============================================================================================================
# Insurance Cost Prediction
#=============================================================================================================

# Import the necessary libraries 
import pandas as pd 
import numpy as np 
import joblib
import json 
from pathlib import Path 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import xgboost as xgb
from dotenv import load_dotenv


# Load the .env variables and define the base working directory 
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the path for config and load the configuration variables from the json files
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Define the path for the clean dataset and the folder to store the trained model
CLEAN_PATH = BASE_DIR/config["clean_data_folder"]
MODEL_PATH = BASE_DIR/config["model_path"]

# Check if the path to store the model exists, if missing , create one
MODEL_PATH.mkdir(parents = True, exist_ok = True )

# Create a full path in the model folder to store the trained model 
SAVE_MODEL = MODEL_PATH/"model.pkl"

def train_model(CLEAN_PATH: Path):
    """ 
    Function to train the model:
        - Load the cleaned dataset
        - Extract independent and dependent variables 
        - Split the dataset into train and test set
        - Build and train the model
        - Inference
        - Evaluation
    """
    # Load the cleaned dataset
    clean_data = pd.read_csv(CLEAN_PATH)

    # Extract the independent(y) and dependent variables(x)
    x = clean_data.iloc[:,:-1].values
    y = clean_data.iloc[:,-1].values

    # Split the cleaned dataset into train and test sets 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    # Build and train the model 
    model = xgb.XGBRegressor(**config["xgb_params"])
    model.fit(x_train, y_train)

    # Inference 
    y_pred = model.predict(x_test)

    # Display the predictions and the true values side by side for comparison 
    np.set_printoptions(precision = 2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    # Evaluation 
    ## R-Squared
    r2 = r2_score(y_test, y_pred)

    ## Adjusted R-Squared 
    k = x_test.shape[1]
    n = x_test.shape[0]

    adj_r2 = 1 - (1 - r2) * (n - 1)/(n - k - 1)
    
    # k-fold cross validation
    avg_r2 = cross_val_score(estimator = model,
                          X = x, 
                          y = y, 
                          scoring = "r2",
                          cv = 10
                         )
    print(" ")
    print(f" Average R-Squared(10-fold):{avg_r2.mean():.3f}")
    print(f"Standard Deviation:{avg_r2.std():.3f}")

    # Return the trained model with its performance metrices 
    return model, r2, adj_r2, avg_r2


if __file__ == "__main__":
    """
    Single entry point as the only place to print the output 
        - Call the function to build and train the model
        - Save the trained mdoel to the full path created
        - Save the perfomance metrics to a json file 
    """

    # Call the fucntion to build and train the model 
    model,r2, adj_r2, avg_r2 = train_model(CLEAN_PATH)

    # Save the model to the model folder 
    joblib.dump(model, SAVE_MODEL)
    
   
    # Save model performance metrics as json
    metrics = {
            "R-Squared": r2,
            "Adjusted R-Squared": adj_r2,
            "Average R-Squared (10-fold)": avg_r2.mean(),
            "Standard Deviation": avg_r2.std()
             }
    # Create a full path in the model folder to store the model metrics 
    metrics_path = MODEL_PATH/"model_metrics.json"

    # Write the model performance metrics to json files 
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent = 4)
   
    
    # Summary 
    print("\n=======================================================================================")
    print(f"The trained model is saved at: {SAVE_MODEL}")
    print(f" Model performance metrics saved at: {metrics_path}")
    print("=======================================================================================\n")


















































































































