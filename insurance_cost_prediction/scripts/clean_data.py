
# Import required libraries 
import pandas as pd 
import numpy as np 
import json 
import os 
from pathlib import Path 

# Define the base working directory 
BASE_DIR = Path(__file__).resolve().parent.parent

# Identify the path to config and load its variables 
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Identify the path to where the raw dataset is stored and the path also to the folder that will store the clean dataset
RAW_DATASET_PATH = (BASE_DIR/config["raw_path"]).resolve()
CLEAN_DATASET = (BASE_DIR/config["clean_folder"]).resolve()

# Check if the clean dataset folder exists, if missing create one automatically 
CLEAN_DATASET.mkdir(parents = True, exist_ok = True)

# Create a full path in the dataset folder to store the processed dataset
PROCESSED_DATASET_PATH = CLEAN_DATASET/"insurance_cleaned.csv"

# Check if the raw dataset file exists
if not RAW_DATASET_PATH.exists():
    raise FileNotFoundError(f"Raw dataset missing at : {RAW_DATASET_PATH}")

def clean_data(raw_dataset_path: Path):
    """
    Function to clean the raw dataset
        - Load the raw dataset
        - Encode the binary columns 
        - One-hot encoding 
        - Return the output
    """

    # Load the raw dataset
    clean_data = pd.read_csv(raw_dataset_path)

    # Encode the binary columns 
    clean_data["sex"] = clean_data["sex"].apply(lambda m: 0 if m == "female" else 1)
    clean_data["smoker"] = clean_data["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # Applying one-hot encoding to the region column to create new binary columns and avoiding the dummy variable trap by dropping one column
    region_dummies = pd.get_dummies(clean_data["region"], drop_first = True, dtype = int)
    clean_data = pd.concat([region_dummies, clean_data], axis = 1)
    clean_data.drop(["region"], axis = 1, inplace = True)

    # Return the clean dataset
    return clean_data


if __name__ =="__main__":
    """
    A single entry point to act as the main execution block for this script
        - Call the function to clean the dataset
        - Save the clean dataset to the folder 
        - Display outputs
    """
    # Call the function to clean the raw dataset 
    processed_dataset = clean_data(RAW_DATASET_PATH)

    # Save the clean dataset to the folder 
    processed_dataset.to_csv(PROCESSED_DATASET_PATH, index = False)

    # Display the results 


    print("\n=========================================================================================")
    print(processed_dataset.sample(5))
    print(" ")
    print(f"The processed dataset is saved at : {PROCESSED_DATASET_PATH}")
    print("=========================================================================================\n")
