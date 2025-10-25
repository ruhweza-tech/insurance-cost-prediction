
# Import necessary libraries 
import pandas as pd 
import os 
import json
import numpy as np 
from dotenv import load_dotenv
from pathlib import Path


# Load the environmental variables and define the base directory of the project
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Build the configuration path, open the json config files and load them as python dictionary
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Add the dataset and folder paths to the base directory
RAW_PATH = BASE_DIR/config["raw_path"]
CLEAN_FOLDER = BASE_DIR/config["clean_data_folder"]

# Check if the CLEAN_FOLDER exists, if missing create one
CLEAN_FOLDER.   mkdir(parents = True , exist_ok = True)

# Create a full path in the CLEAN_FOLDER to store the cleaned dataset
SAVE_PATH = CLEAN_FOLDER/"insurance_clean.csv"

def clean_data(raw_path: Path):
    """ 
    Function is to clean the raw dataset
        - Load the raw dataset
        - Encode the binary columns 
        - One - hot encode 
        - Return the clean dataset 
    """

    # Load the raw dataset 
    clean_dataset = pd.read_csv(raw_path)

    # Encode the binary columns
    clean_dataset["sex"] = clean_dataset["sex"].apply(lambda m: 0 if m == "female" else 1)
    clean_dataset["smoker"] = clean_dataset["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # One - hot encode the region column
    region_dummies = pd.get_dummies(clean_dataset["region"], drop_first = True, dtype = int)
    clean_dataset = pd.concat([region_dummies, clean_dataset], axis = 1)
    clean_dataset.drop(["region"], axis = 1, inplace = True)

    return clean_dataset


""" 
Create a single entry point for printing out the results only when run in this scrip. Should not run when script is imported
"""

if __name__ == "__main__":

    # Call the function to clean and save the dataset to the path created
    clean_dataset = clean_data(RAW_PATH)
    clean_dataset.to_csv(SAVE_PATH, index = False)

    print(clean_dataset.head())
    print(" ")
    print(f"\n The cleaned dataset is saved at: {SAVE_PATH}\n")
    print(f"Environment:{os.getenv('ENVIRONMENT', 'local')}")
    print(f"Author: {os.getenv('AUTHOR', 'Mugisha')}")

