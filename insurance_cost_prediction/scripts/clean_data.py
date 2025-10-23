
# Import necessary libraries 
import pandas as pd 
import numpy as np 
import os 
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
config_path = BASE_DIR/"config"/"insurance_config.json"

with open(config_path) as f:
    config = json.load(f)


raw_path = BASE_DIR/ config["raw_path"]
clean_folder = BASE_DIR/ config["clean_data_folder"]
clean_folder.mkdir(parents = True, exist_ok = True)
save_path = clean_folder/ "insurance_clean.csv"


def clean_data(raw_path):

#===================================================================================
# Function to clean the dataset
    # Description:
    #   - Load the dataset
    #   - Encode the binary columns 
    #   - One-hot encode the region columns 
#===================================================================================

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


#===================================================================================
# Main execution block
    # Purpose:
    #   Ensures that this script runs directly here when executed not when imported
#===================================================================================
if __name__ == "__main__":

    # Call the fucntion to clean the raw dataset
    dataset = clean_data(raw_path)

    # Save the cleaned dataset to the full path created 
    dataset.to_csv(save_path, index = False)

    print(dataset.sample(5))
    print(f"\nThe cleaned dataset is saved at: {save_path}")

