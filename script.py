# Installs and imports 

#!pip install transformers torch
#!pip install bs4
#!pip install tf-keras
#!pip install pydub

import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
import torch
import sacrebleu


import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json
from tqdm import tqdm
import tarfile
import os
import json

base_url = "https://huggingface.co/datasets/google/fleurs-r/resolve/main/"
LANG_CODE = "fr_fr"  

def get_folder_names(language_code=LANG_CODE):
    return [
        f"data/{language_code}/audio/dev.tar.gz",
        f"data/{language_code}/audio/test.tar.gz",
        f"data/{language_code}/audio/train.tar.gz"
    ]

def get_fleurs_data(file_path, language_code=LANG_CODE):  
    output_dir = f"fleurs_{language_code}_audio"  
    os.makedirs(output_dir, exist_ok=True)  # This will create a single directory for all extracted files

    file_url = base_url + file_path  # Correct URL format
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


def get_fleurs_file_codes(language_code=LANG_CODE):
    file_paths = {
        "dev" : base_url+f"data/{language_code}/dev.tsv", 
        "test": base_url+f"data/{language_code}/test.tsv",
        "train": base_url+f"data/{language_code}/train.tsv"
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


get_fleurs_file_codes()

french_file_names = get_folder_names(LANG_CODE)
for file_path in french_file_names:
    get_fleurs_data(file_path, LANG_CODE)

def get_whisper(model='openai/whisper-large-v3'):
    whisper = pipeline("automatic-speech-recognition", model, torch_dtype=torch.float16, device="cuda:0")
    return whisper

whisper = get_whisper("openai/whisper-large-v3")  # defaulting to small for now


def get_transcription(audio_file):
    transcription = whisper(audio_file, return_timestamps=True)
    return transcription['text']



def parse_to_whisper(ds, langage = "fr"):
    with open(f"{language}_whisper_out.txt", "a") as output_file:
        for example in ds:
            tr = get_fleurs_data_fr(example)
            output_file.write(tr + "\n")



def get_translation(text, source_language, target_language):
    # Get the translation from the sample text
    
    url = f"https://translate.google.com/m?tl={target_language}&sl={source_language}&q={text}"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}")

    # Parse the page content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the translated text from the page
    translation = soup.find('div', class_='result-container').text

    return translation


def transcribe_dataset(ds):
    output_dir = f"fleurs_{LANG_CODE}_audio/{ds}"  # Path to directory
    output_file = f"{LANG_CODE}_whisper_out_large.json"

    results = []  # Store results as a list of dictionaries

    for file_name in os.listdir(output_dir):  # List files in the directory
        if file_name.endswith(".wav"):  # Assuming audio files are .wav (adjust if needed)
            file_path = os.path.join(output_dir, file_name)  # Full path to the file
            transcript = get_transcription(audio_file=file_path)  # Pass the full path
            print(transcript)
            results.append({
                "file_name": file_name,
                "transcript": transcript,
            })
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


def translate_dataset(source_language="fr", target_language="en", ds=None):
    with open(ds, 'r', encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        if "transcript" in sample and "translation" not in sample:  # Avoid redundant translation
            sample["translation"] = get_translation(sample["transcript"], source_language, target_language)

    with open(ds, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




    


def parser_txt(source_language="fr", target_language="en", ds=None):                    # easier to follow the output but not best format
    output_dir = f"fleurs_{LANG_CODE}_audio/{ds}"                                       # path to directory
    with open(f"{source_language}_whisper_out.txt", "a") as output_file:
        for file_name in os.listdir(output_dir):  # List files in the directory
            if file_name.endswith(".wav"):  # Assuming audio files are .wav (adjust if needed)
                file_path = os.path.join(output_dir, file_name)  # Full path to the file
                transcript = get_transcription(audio_file=file_path)  # Pass the full path
                translation = get_translation(transcript, source_language, target_language)
                output_file.write(file_name + ": Transcript = " + transcript + "    Translation = " + translation + "\n")


def parser_json(source_language="fr", target_language="en", ds=None):
    output_dir = f"fleurs_{LANG_CODE}_audio/{ds}"  # Path to directory
    output_file = f"{source_language}_whisper_out.json"

    results = []  # Store results as a list of dictionaries

    for file_name in os.listdir(output_dir):  # List files in the directory
        if file_name.endswith(".wav"):  # Adjust if needed for different file types
            file_path = os.path.join(output_dir, file_name)  # Full path to the file
            transcript = get_transcription(audio_file=file_path)  # Get transcript
            translation = get_translation(transcript, source_language, target_language)  # Get translation
            
            # Store result in a dictionary
            results.append({
                "file_name": file_name,
                "transcript": transcript,
                "translation": translation
            })

    # Write results to a JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


dtt = ["dev", "test", "train"]
transcribe_dataset(ds="dev")

translate_dataset(source_language="fr", target_language="en", ds="fr_fr_whisper_out_large.json")


path = "https://huggingface.co/datasets/openlanguagedata/flores_plus/blob/main/devtest/fra_Latn.parquet"
print(path)
df_parquet = pd.read_parquet("fra_Latn-4.parquet")
print(df_parquet.columns)
