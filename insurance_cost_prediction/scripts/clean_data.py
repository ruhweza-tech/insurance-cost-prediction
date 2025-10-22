
# Import necessary libraries 
import pandas as pd 
import numpy as np 
import os 


# Define the path of the raw dataset 
raw_path = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\data\raw\insurance.csv"

# Create a function to clean the data 
def clean_data(raw_path):

    # Load dataset 
    dataset = pd.read_csv(raw_path)

    # Encode binary columns 
    dataset["sex"] = dataset["sex"].apply(lambda m : 0 if m == "female" else 1)
    dataset["smoker"] = dataset["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # One - hot encode region column 
    region_dumies = pd.get_dummies(dataset["region"], drop_first = True, dtype = int)
    dataset = pd.concat([region_dumies, dataset], axis = 1)
    dataset.drop(["region"], axis = 1, inplace = True)
    
    return dataset

if __name__ == "__main__":

    # Define the folder where you want to store the cleaned dataset
    folder = r"C:\Users\ruhwemug\ML_Projects\my_project\insurance_cost_prediction\data\clean"

    # Check if the folder exists,  if its missing create one 
    os.makedirs(folder, exist_ok = True)

    # Create a full path in this folder to save the cleaned dataset
    clean_path = os.path.join(folder, "insurance_cleaned.csv")

    # call the functiuon to the clean the dataset 
    data_cleaned = clean_data(raw_path)

    # Store the cleaned dataset to the full path created 
    data_cleaned.to_csv(clean_path, index = False)
    print(data_cleaned.head())



