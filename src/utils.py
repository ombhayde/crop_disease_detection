# src/utils.py
import os
import requests
import zipfile

def download_dataset():
    """
    Downloads the PlantVillage dataset from Kaggle
    Note: You'll need a Kaggle account and API key
    """
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    print("To download the PlantVillage dataset:")
    print("1. Go to https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("2. Download the dataset and extract to data/raw/")
    print("3. Or use Kaggle API: kaggle datasets download -d abdallahalidev/plantvillage-dataset")
    
    # Alternative: smaller dataset for testing
    url = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    print(f"Alternatively, downloading smaller rice disease dataset for testing from {url}")
    
    response = requests.get(url)
    with open('data/raw/rice_disease_dataset.zip', 'wb') as f:
        f.write(response.content)
    
    # Extract the zip file
    with zipfile.ZipFile('data/raw/rice_disease_dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('data/raw/')