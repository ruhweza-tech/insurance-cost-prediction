
# Import required libraries 
import os 
import json 
import pandas as pd 
from pathlib import Path 
from dotenv import load_dotenv

# Load the .env variables and define the base working directory 
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the path to config and load the configuration variables from json
CONFIG_PATH = BASE_DIR/"config"/"insurance_config.json"
with open(CONFIG_PATH) as m:
    config = json.load(m)

# Define the path to the raw dataset and the path to the folder that will store the cleaned dataset
RAW_PATH = BASE_DIR/config["raw_path"]
CLEAN_FOLDER = BASE_DIR/config["clean_data_folder"]

# Check if the clean folder exists , if missing create one 
CLEAN_FOLDER.mkdir(parents = True, exist_ok = True)

# Create a full path in the clean folder to store the cleaned dataset
SAVE_PATH = CLEAN_FOLDER/"insurance_cleaned.csv"

def clean_data(raw_data: Path):
    """
    Function to clean the raw dataset
        - Load the raw dataset
        - Encode the binary columns 
        - One-hot encoding 
        - Return the clean_dataset
    """
    # Load the raw dataset 
    clean_dataset = pd.read_csv(raw_data)

    # Encode the binary categorical variables
    clean_dataset["sex"] = clean_dataset["sex"].apply(lambda m: 0 if m == "female" else 1)
    clean_dataset["smoker"] = clean_dataset["smoker"].apply(lambda m: 0 if m == "no" else 1)

    # One-hot encoding the region column and droping the first category to prevent dummy trap
    region_dummies = pd.get_dummies(clean_dataset["region"], drop_first = True, dtype = int)
    clean_dataset = pd.concat([region_dummies, clean_dataset], axis = 1)
    clean_dataset.drop(["region"], axis = 1, inplace = True)

    return clean_dataset

if __name__ == "__main__":
    """
    Single entry point as the main execution block
        - Call the function to clean the raw dataset
        - Save the cleaned dataset to the full path already created earlier on
        - Display a message indicating where the cleaned dataset is saved
    """
    # Call the fucntion to clean the raw dataset
    clean_dataset = clean_data(RAW_PATH)

    # Save the cleaned dataset to the clean folder path
    clean_dataset.to_csv(SAVE_PATH, index = False)

    # Display the output

    print("\n====================================================================================")
    print(clean_dataset.head())
    print(" ")
    print(f"The cleaned dataset is saved at: {SAVE_PATH}")
    print("====================================================================================\n")

    print("\n====================================================================================")
    print(f"Environment is : {os.getenv("ENVIRONMENT", "local")}")
    print(f"The author for this script is called: {os.getenv("AUTHOR", "Mugisha")}")
    print("====================================================================================\n")
    