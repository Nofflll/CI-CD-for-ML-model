import os
import argparse
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data(output_folder):
    """
    Downloads the wine quality dataset and saves it to the specified folder.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        logging.info(f"Downloading data from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        file_path = os.path.join(output_folder, "winequality-red.csv")
        
        with open(file_path, "wb") as f:
            f.write(response.content)
            
        logging.info(f"Data downloaded and saved to {file_path}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the downloaded data.")
    
    args = parser.parse_args()
    
    get_data(args.output_folder) 