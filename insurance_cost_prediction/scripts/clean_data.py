
# Import necessary libraries 
import pandas as pd 
import numpy as np 
import os 
import joblib

# Define the path for the raw dataset 
raw_path = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\data\raw\insurance.csv"

# Create a function to clean the raw dataset 
def clean_data(raw_path):

    # Load the dataset
    dataset = pd.read_csv(raw_path)

    # Encode the binary columns 
    dataset["sex"] = dataset["sex"].apply(lambda m: 0 if m == "female" else 1)
    dataset["smoker"] = dataset["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # One-hot encode the region columns 
    region_dummies = pd.get_dummies(dataset["region"], drop_first = True , dtype = int)
    dataset = pd.concat([region_dummies, dataset], axis = 1)
    dataset.drop(["region"], axis = 1, inplace = True)

    return dataset


# Create a single entry point to run the code 
if __name__ == "__main__":

    # Define the folder path to store the cleaned dataset 
    folder = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\data\clean"

    # Check if the folder path exists , if missing create one
    os.makedirs(folder, exist_ok = True)

    # Create a full path in the folder to store the cleaned dataset 
    save_path = os.path.join(folder, "insurance_cleaned.csv")

    # Call the fucntion to clean the raw dataset
    dataset = clean_data(raw_path)

    # Save the cleaned dataset to the full path created 
    dataset.to_csv(save_path, index = False)

    print(dataset.sample(5))
    print(f"\nThe cleaned dataset is saved at: {save_path}")

