
# Import necessary libraries 
import pandas as pd 
import numpy as np 
import os 
import json 
from pathlib import Path
from dotenv import load_dotenv

# Load the .env variables and define the base working directory
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the configuration path and read the configuration variables from the json files 
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Define the path to the raw data and the folder to store the clean dataset
RAW_PATH = BASE_DIR/config["raw_path"]
CLEAN_FOLDER = BASE_DIR/config["clean_data_folder"]

# Check if the clean folder exists, if missing create one 
CLEAN_FOLDER.mkdir(parents = True, exist_ok = True)

# Create a full path in the clean folder to store the cleaned dataset 
SAVE_PATH = CLEAN_FOLDER/"insurance_clean.csv"

def clean_data(raw_path: Path):
    """ 
    Function to clean the raw dataset:
        - Load raw dataset
        - Encode binary columns 
        - One-hot encoding 
        - Return clean dataset
    """

    # Load raw dataset
    clean_dataset = pd.read_csv(raw_path)

    # Encode binary columns 
    clean_dataset["sex"] = clean_dataset["sex"].apply(lambda m: 0 if m == "female" else 1)
    clean_dataset["smoker"] = clean_dataset["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # One - hot encoding the region column 
    region_dummies = pd.get_dummies(clean_dataset["region"], drop_first = True, dtype = int)
    clean_dataset = pd.concat([region_dummies, clean_dataset], axis = 1)
    clean_dataset.drop(["region"], axis = 1, inplace = True)


    return clean_dataset


if __name__ == "__main__":
    """ 
    Single entry point to make sure the code is just printed out in this script not when it is imported 
        - Call the function to clean and then save the clean dataset
    """
    # Call the function to clean the dataset 
    clean_dataset = clean_data(RAW_PATH)

    # Save the cleaned dataset to the full path created in the CLEAN_FOLDER
    clean_dataset.to_csv(SAVE_PATH, index = False)

    # Display the output 
    print(clean_dataset.head(5))

    print(" ")

    print(f"The cleaned dataset is saved at {SAVE_PATH}\n")
    print(f"Environment:{os.getenv("ENVIRONMENT", "local")}")
    print(f"AUTHOR:{os.getenv("AUTHOR", "Mugisha")}")



