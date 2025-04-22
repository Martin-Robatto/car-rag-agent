import kagglehub
import pandas as pd
import os

def download_dataset():
    # This will download the latest dataset version
    path = kagglehub.dataset_download("tawfikelmetwally/automobile-dataset")
    print("Dataset downloaded to:", path)

    # Find a CSV file in the directory
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_path = os.path.join(path, file)
            df = pd.read_csv(csv_path)
            print(f"Loaded: {file}")
            return df

    raise FileNotFoundError("No CSV file found in dataset.")

# Global variable to store the dataset
automobile_df = None

def get_dataset():
    """
    Returns the automobile dataset as a pandas DataFrame.
    Downloads it if not already downloaded.
    
    Returns:
        pandas DataFrame containing the automobile dataset
    """
    global automobile_df
    if automobile_df is None:
        automobile_df = download_dataset()
    return automobile_df

if __name__ == "__main__":
    df = download_dataset()
    print(df.head())
