
# Import necessary libraries 
import os 
import json 
import pandas as pd 
import numpy as np 
from dotenv import load_dotenv
from pathlib import Path 

# Load the .env variables and define the base directory 
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the config path and load json variables 
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Resolve the json files
RAW_PATH = BASE_DIR/config["raw_path"]
CLEAN_FOLDER = BASE_DIR/config["clean_data_folder"]

# Check if the CLEAN_FOLDER path exists, if missing create one 
CLEAN_FOLDER.mkdir(parents = True, exist_ok = True)

# Create a full path in the CLEAN_FOLDER to store the clean datatset
SAVE_PATH = CLEAN_FOLDER/"insurance_clean.csv"


def clean_data(raw_path: Path):
    #==========================================================================
    # FUNCTION: To clean raw dataset
    #   - Load raw dataset
    #   - Encode binary columns 
    #   - One-hot encode 
    #   - Return clean dataset

    # Load raw dataset
    clean_dataset = pd.read_csv(raw_path)

    # Encode binary columns 
    clean_dataset["sex"] = clean_dataset["sex"].apply(lambda m: 0 if m == "female" else 1)
    clean_dataset["smoker"] = clean_dataset["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # one-hot encode region columns 
    region_dummies = pd.get_dummies(clean_dataset["region"], drop_first = True, dtype = int)
    clean_dataset = pd.concat([region_dummies, clean_dataset], axis = 1)
    clean_dataset.drop(["region"], axis = 1, inplace = True)

    return clean_dataset


# Create a single entry point to run the script directly here so that it is  not run when imported
if __name__ == "__main__":

    # Call the fucntion to clean the raw dataset and save the cleaned dataset to the SAVE_PATH 
    clean_dataset = clean_data(RAW_PATH)
    clean_dataset.to_csv(SAVE_PATH, index = False)

    print(clean_dataset.head())
    print(f"\nThe cleaned dataset is saved at: {SAVE_PATH}\n")
    print(f"Enviroment:{os.getenv('ENVIROMENT','local')}")
    print(f"AUTHOR: {os.getenv('AUTHOR', 'Mugisha')}")

