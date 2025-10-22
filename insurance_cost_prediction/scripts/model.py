#=============================================================================================================
# Insurance Cost Prediction
#=============================================================================================================


# import necessary libraries 
import joblib
import os 
import json
import pandas as pd 
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score


# Define the path for the cleaned dataset 
clean_path = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\data\clean\insurance_cleaned.csv"

# Creae a function to build and train the model 

def train_model(clean_path):

    # Load the cleaned dataset
    dataset = pd.read_csv(clean_path)

    # Extract the independent and dependent features 
    x = dataset.iloc[:,:-1].values 
    y = dataset.iloc[:,-1].values 

    # Split the dataset into train and test set 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    # Build and train the model 
    model = xgb.XGBRegressor(max_depth = 2, learning_rate = 0.1, n_estimators = 100, random_state = 0)
    model.fit(x_train, y_train)

    # Make predictions on the dataset
    y_pred = model.predict(x_test)
    np.set_printoptions(precision = 2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    # R-Squared 
    r2 = r2_score(y_test, y_pred)

    # Adjusted R-squared 
    k = x_test.shape[1]
    n = x_test.shape[0]

    adj_r2 = 1 - (1 - r2)*(n - 1)/(n - k - 1)

    # K-fold cross validation 
    r2s = cross_val_score(
                        estimator = model ,
                        X = x,
                        y = y,
                        scoring = "r2",
                        cv = 10
                        )
    print("")
    print("K-Fold Cross Validation Average R-Squared: {:.3f}".format(r2s.mean()))
    print("Standard Deviation: {:.3f}".format(r2s.std()))

    return model, r2, adj_r2, r2s


if __name__ == "__main__":

    # Define the folder path where you want to store the model
    folder = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\model"

    # Check if the folder exists or create one if missing 
    os.makedirs(folder, exist_ok = True)

    # Create a full path in the folder to store the model
    model_path = os.path.join(folder, "model.pkl")

    # Call the fucntion to train the model 
    model,r2, adj_r2, r2s = train_model(clean_path)

    # Save the trained model to the path 
    joblib.dump(model, model_path)


    # save model metrics to json file 
    metrics = {
                "R-Squared": r2,
                "Adjusted R-Squared": adj_r2,
                "K-Fold Cross Validation R-Squared":r2s.mean(),
                "Standard Deviation": r2s.std()
            }
    
    # create a full path to store the json file containing the model metrics
    metrics_path = os.path.join(folder, "model_metrics.json")

    # write metrics to json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent = 4)

    print(f"Model metrics successfully saved at: {metrics_path}")