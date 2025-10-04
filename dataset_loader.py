import pandas as pd
import os

def load_dataset(dataset_type="train", folder="dataset"):
    if dataset_type not in ["train", "test", "sample_submission"]:
        raise ValueError("dataset_type must be 'train' or 'test'")
    
    file_path = os.path.join(folder, f"{dataset_type}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    
    df = pd.read_csv(file_path)

    return df
