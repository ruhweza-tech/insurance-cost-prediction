
# Import the required libraries
import pandas as pd 
import json 
from pathlib import Path 
import numpy as np 

# Define the working base directory [2 levels up]
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the path to the config and load the variables from json
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Define the path to the raw dataset and the folder to store the cleaned dataset
RAW_PATH = BASE_DIR/config["raw_path"]
CLEAN_PATH = BASE_DIR/config["clean_data_folder"]

# Check if the folder to store the clean dataset exists, if missing , automatically create one
CLEAN_PATH.mkdir(parents = True, exist_ok = True)

# Create a full path in the folder to store the cleaned dataset
PROCESSED = CLEAN_PATH/"insurance_cleaned.csv"

def clean_data(raw_path: Path):
    """ 
    Function to clean the raw dataset
        - Load the raw dataset
        - Encode binary columns 
        - One-hot encode 
        - Return the output
    """
    # Load the raw dataset
    raw_dataset = pd.read_csv(raw_path)

    # Encode binary columns 
    raw_dataset["sex"] = raw_dataset["sex"].apply(lambda m: 0 if m == "female" else 1)
    raw_dataset["smoker"] = raw_dataset["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # One-hot encode region column
    region_dummies = pd.get_dummies(raw_dataset["region"], drop_first = True, dtype = int)
    raw_dataset = pd.concat([region_dummies, raw_dataset], axis = 1)
    raw_dataset.drop(["region"], axis = 1, inplace = True)

    # Retrun the output
    return raw_dataset


if __name__ == "__main__":
    """
    Single entry point to act is the mian execution block
        - Call the fucntion to clean the raw dataset
        - Store the cleaned dataset in the folder
        - Display the output
    """
    # Call the fucntion to clean the dataset
    processed_data = clean_data(RAW_PATH)

    # Store the cleaned dataset in the folder
    processed_data.to_csv(PROCESSED, index = False)

    print("\n=================================================================================")
    print(processed_data.sample(5))
    print(" ")
    print(f"The cleaned dataset is saved at: {PROCESSED}")
    print("=================================================================================\n")