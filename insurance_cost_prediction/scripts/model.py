#=============================================================================================================
# Insurance Cost Prediction
#=============================================================================================================

# Import Required Librarries
import os 
import joblib 
import json
import pandas as pd 
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score


# Path to the Cleaned Dataset (CSV)
clean_path = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\data\clean\insurance_cleaned.csv"


def train_model(clean_path):

    #=================================================================
    # Function to train the dataset
    # Description:
    #       - Load cleaned dataset
    #       - Split the dataset into train and test sets 
    #       - Build and train the model using xgboost regression
    #       - Evaluation (R², Adjusted R², K-Fold Cross Validation)
    #       - Return trained model and performance metrics
    #=================================================================


    # Load the clean datatset
    dataset = pd.read_csv(clean_path)

    # Extract independent and dependent features from the dataset
    x = dataset.iloc[:,:-1].values 
    y = dataset.iloc[:,-1].values 

    # Split the dataset into train and test sets 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    # Build and train the model
    model = xgb.XGBRegressor(max_depth = 2, learning_rate = 0.1, n_estimators = 100, random_state = 0)
    model.fit(x_train, y_train)

    # Make predictions on the dataset 
    y_pred = model.predict(x_test)

    # Display the predictions and the true values side by side for comparision
    print(" ")
    np.set_printoptions(precision = 2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    # Compute R-Squared 
    r2 = r2_score(y_test, y_pred)

    # Compute Adjusted R-Squared
    k = x_test.shape[1]
    n = x_test.shape[0]

    adj_r2 = 1 - (1 - r2) * (n - 1)/(n - k - 1)

    # Perform 10-fold cross validation for model stability
    avg_r2 = cross_val_score(
                            estimator = model,
                            X = x, 
                            y = y,
                            scoring = "r2",
                            cv = 10
                            )
    print(" ")
    print("Average R-Squared (10-fold): {:.3f}".format(avg_r2.mean()))
    print("Standard Deviation: {:.3f}".format(avg_r2.std()))

    # Return trained model and performance metrics
    return model, r2, adj_r2, avg_r2


#===================================================================================
# Main execution block
    # Purpose:
    #   Ensures that this script runs directly here when executed not when imported
#===================================================================================

if __name__ == "__main__":

    # Define the folder path to store the trained model
    model_path = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\model"

    # Check if the path exits, if missing create one 
    os.makedirs(model_path, exist_ok = True)

    # Create a full path in the model_path to store the trained model 
    save_model = os.path.join(model_path, "model.pkl")

    # Call the function to train the model and calculate model metrics 
    model, r2, adj_r2, avg_r2 = train_model(clean_path)

    # Save the trained model to the full path created 
    joblib.dump(model, save_model)

    # Collect model performace metrics
    metrics = {
                "R-Sqaured":r2,
                "Adjusted R-Squared": adj_r2,
                "Average R-Squared (10-fold cv)": avg_r2.mean(),
                "Standard Deviation": avg_r2.std()
                }
    
    # Create a full path to store the model metrics 
    metrics_path = os.path.join(model_path, "metrics.json")

    # Write the model metrics to json files and store to the path
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent = 4)

    print(" ")
    print(f"The trained model and model metrics are all sucessfully stored at: {save_model}")