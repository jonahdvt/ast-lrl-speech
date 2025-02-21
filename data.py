from config import FLEURS_BASE_URL

import pandas as pd
import requests
import tarfile
import os



def get_folder_names(language_code=None):
    return [
        f"data/{language_code}/audio/dev.tar.gz",
        f"data/{language_code}/audio/test.tar.gz",
        f"data/{language_code}/audio/train.tar.gz"
    ]

def get_fleurs_data(file_path, language_code=None):  
    output_dir = f"fleurs_{language_code}_audio"  
    os.makedirs(output_dir, exist_ok=True)  # This will create a single directory for all extracted files

    file_url = FLEURS_BASE_URL + file_path  # Correct URL format
    local_filename = os.path.basename(file_path)  
    local_path = os.path.join(output_dir, local_filename)

    print(f"Downloading {file_url}...")
    response = requests.get(file_url, stream=True)

    if response.status_code != 200:
        print(f"Failed to download {file_url}, Status Code: {response.status_code}")
        return  

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the tar.gz file in the same directory
    with tarfile.open(local_path, "r:gz") as tar:  
        tar.extractall(output_dir)  # All files go into the same output_dir

    print(f"Extracted: {local_filename}")
    os.remove(local_path)  # Remove the tar.gz file after extraction


def get_fleurs_file_codes(language_code=None):
    file_paths = {
        "dev" : FLEURS_BASE_URL+f"data/{language_code}/dev.tsv", 
        "test": FLEURS_BASE_URL+f"data/{language_code}/test.tsv",
        "train": FLEURS_BASE_URL+f"data/{language_code}/train.tsv"
        }

    output_combined_csv = f"{language_code}_codes_file.csv"

    all_data = []

    for split, path in file_paths.items():
        output_csv = f"data/{language_code}_{split}.csv"  # Separate CSV for each dataset
        df = pd.read_csv(path, sep="\t", encoding="utf-8", usecols=[0,1])  # Read only the first 2 columns
        df.columns = ['codes', 'wav_codes']  # Rename columns to 'codes' and 'wav_codes'

        all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)  # Combine all data into one DataFrame
        combined_df.to_csv(output_combined_csv, index=False, encoding="utf-8")  # Save as a single CSV
        return combined_df







def get_gold_transcript(src_lang=None):
    en_fleurs_url = "https://huggingface.co/datasets/google/fleurs/raw/main/"
    file_paths = {
        "dev": en_fleurs_url + f"data/{src_lang}/dev.tsv", 
        "test": en_fleurs_url + f"data/{src_lang}/test.tsv",
        "train": en_fleurs_url + f"data/{src_lang}/train.tsv"
    }

    output_combined_csv = f"{src_lang}_fleurs_transcript_file.csv"
    all_data = []

    for split, path in file_paths.items():
        df = pd.read_csv(path, sep="\t", encoding="utf-8", on_bad_lines="skip")
        expected_columns = ["codes", "wav_codes", "transcript", "normalized_transcript", "tokens", "duration", "gender"]
        if df.shape[1] != len(expected_columns):
            print(f"Unexpected column count in {path}. Found {df.shape[1]}, expected {len(expected_columns)}. Skipping file.")
            continue

        df.columns = expected_columns 
        # Keep only the first three columns
        df = df[["codes", "wav_codes", "transcript"]]
        all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)  # Combine all data into one DataFrame
        combined_df.to_csv(output_combined_csv, index=False, encoding="utf-8")  # Save as a single CSV
        return combined_df


def get_gold_translation():
    en_fleurs_url = "https://huggingface.co/datasets/google/fleurs/raw/main/"
    file_paths = {
        "dev" : en_fleurs_url+f"data/en_us/dev.tsv", 
        "test": en_fleurs_url+f"data/en_us/test.tsv",
        "train": en_fleurs_url+f"data/en_us/train.tsv"
        }

    output_combined_csv = "en_translations_file.csv"

    all_data = []

    for split, path in file_paths.items():
        df = pd.read_csv(path, sep="\t", encoding="utf-8", on_bad_lines="skip")
        expected_columns = ["codes", "wav_codes", "transcript", "normalized_transcript", "tokens", "duration", "gender"]
        if df.shape[1] != len(expected_columns):
            print(f"Unexpected column count in {path}. Found {df.shape[1]}, expected {len(expected_columns)}. Skipping file.")
            continue

        df.columns = expected_columns 
        all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)  # Combine all data into one DataFrame
        combined_df.to_csv(output_combined_csv, index=False, encoding="utf-8")  # Save as a single CSV
        return combined_df
    
