import os
import json

# Get the directory where your script is located
current_directory = os.path.dirname(__file__)

# Define the relative path to your JSON file
relative_path = "models/XGB_1&2_semesters_2015_17-02-2024_11-54-21_metadata.json"

# Combine the current directory with the relative path
file_path = os.path.join(current_directory, relative_path)

try:
    with open(file_path, 'r') as f:
        model_meta = f.read()
        if not model_meta:
            print("Error: JSON file is empty")
            # Handle the error as needed
        else:
            model_meta = json.loads(file_path)
            print(model_meta)
            # Continue with your code
except FileNotFoundError:
    print("Error: JSON file not found")
    # Handle the error as needed
except json.decoder.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    # Handle the error as needed