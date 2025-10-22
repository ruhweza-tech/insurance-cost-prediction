
# Import necessary libraries 
import pandas as pd 
import numpy as np 
import os 
import joblib


# Define the path for the raw dataset 

raw_path = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\data\raw\insurance.csv"

# Create a function that will clean the raw dataset
def clean_data(raw_path):

    # Load dataset 
    dataset = pd.read_csv(raw_path)

    # Encode the binary columns 
    dataset["sex"] = dataset["sex"].apply(lambda m: 0 if m == "female" else 1)
    dataset["smoker"] = dataset["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # One hot encode region
    region_dummies = pd.get_dummies(dataset["region"], drop_first = True, dtype = int)
    dataset = pd.concat([region_dummies, dataset], axis = 1)
    dataset.drop(["region"], axis = 1, inplace = True)

    return dataset


# create a single entry point
if __name__ == "__main__":

    # Define the folder path to store the cleaned datatset 
    folder = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\data\clean"

    # Check if the folder exists , if missing create one
    os.makedirs(folder, exist_ok = True )

    # Create a full path in the folder to save the cleaned dataset
    save_path = os.path.join(folder , "insurance_cleaned.csv")

    # Call the function the clean the dataset 
    clean_dataset = clean_data(raw_path)

    # Save the cleaned dataset in the path
    clean_dataset.to_csv(save_path, index = False)

    print(clean_dataset.head())
    print(f"Cleaned dataset saved at: {save_path}")