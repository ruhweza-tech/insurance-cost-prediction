
# Import the necessary libraries 
import pandas as pd 
import numpy as np 
import json
import os
from pathlib import Path 
from dotenv import load_dotenv

# Load the .env variables and define the working base directory for the project 
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the config path and load the configuration variables from the json files 
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Define the path to the raw dataset and that to the folder for storing the cleaned dataset
RAW_PATH = BASE_DIR/config["raw_path"]
CLEAN_FOLDER = BASE_DIR/config["clean_data_folder"]

# Check if the clean folder exists, if missing , create one 
CLEAN_FOLDER.mkdir(parents = True, exist_ok = True)

# Create a full path in the clean folder to store the cleaned dataset
SAVE_PATH = CLEAN_FOLDER/"insurance_cleaned.csv"

def clean_data(raw_path: Path):
    """ 
    Function to clean the raw dataset
        - Load the raw dataset
        - Encode the binary columns 
        - one-hot encoding 
        - Return the clean dataset
    """
    # Load the raw dataset
    clean_dataset = pd.read_csv(raw_path)

    # Encode the binary columns 
    clean_dataset["sex"] = clean_dataset["sex"].apply(lambda m: 0 if m == "female" else 1)
    clean_dataset["smoker"] = clean_dataset["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # One-hot encoding the region column
    region_dummies = pd.get_dummies(clean_dataset["region"], drop_first = True, dtype = int)
    clean_dataset = pd.concat([region_dummies,clean_dataset], axis = 1)
    clean_dataset.drop(["region"], axis = 1, inplace = True)

    return clean_dataset

if __name__ == "__main__":
    """ 
    Single entry point that allows the script to only be printed out here not when importíng the script
        - Call the function to clean the raw dataset
        - Save the cleaned dataset to the path created 
        - Display results 
    """
    # Call the fucntion to clean the raw dataset
    clean_dataset = clean_data(RAW_PATH)

    # Save the cleaned dataset to the path created in the clean folder 
    clean_dataset.to_csv(SAVE_PATH, index = False)
    
    # Display the output 
    print(clean_dataset.sample(5))
    print(f"\nThe cleaned dataset is saved at: {SAVE_PATH}")

    print(" ")
    print(f"Environment:{os.getenv("ENVIRONMENT", "local")}")
    print(f"Author: {os.getenv("AUTHOR", "Mugisha")}")