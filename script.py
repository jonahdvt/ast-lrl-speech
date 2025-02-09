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





def get_whisper(model='openai/whisper-large-v3'):
    whisper = pipeline("automatic-speech-recognition", model, torch_dtype=torch.float16, device="cuda:0")
    return whisper




def get_transcription(whisper, audio_file):
    transcription = whisper(audio_file, return_timestamps=True)
    return transcription['text']



def get_translation(text, source_language, target_language, whisper_model):
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


def transcribe_dataset(ds, whisper_model):
    output_dir = f"fleurs_{LANG_CODE}_audio/{ds}"  # Path to directory
    output_file = f"{LANG_CODE}_whisper_out_large.json"

    results = []  # Store results as a list of dictionaries

    for file_name in os.listdir(output_dir):  # List files in the directory
        if file_name.endswith(".wav"):  # Assuming audio files are .wav (adjust if needed)
            file_path = os.path.join(output_dir, file_name)  # Full path to the file
            transcript = get_transcription(audio_file=file_path, whisper=whisper_model)  # Pass the full path
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
        if "transcript" in sample and "whisper_translation" not in sample:  # Avoid redundant translation
            sample["whisper_translation"] = get_translation(sample["transcript"], source_language, target_language)

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
    


# MATCHING 

def gold_codes_matching(json_path, csv_path, output_path):   # function to get the codes that will match each transcript with their english counterpart from the gold dataset 
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Create a mapping from wav_codes to codes
    name_mapping = dict(zip(df['wav_codes'], df['codes']))

    # Update JSON entries
    for entry in json_data:
        file_name = entry.get('file_name')
        if file_name in name_mapping:
            entry['code'] = name_mapping[file_name]  # Add the matching code

    # Save the updated JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"Updated JSON saved to {output_path}")

def gold_translation_matching(json_path, csv_path, output_path):  # match the audio code and fetch the fleurs transcript in english for corresponding audio
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Create a mapping from codes to transcript in the en csv
    name_mapping = dict(zip(df['codes'], df['transcript']))

    # Update JSON entries
    for entry in json_data:
        file_code = entry.get('code')
        if file_code in name_mapping:
            entry['gold_translation'] = name_mapping[file_code]  # Add the gold translation

    # Save the updated JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"Updated JSON saved to {output_path}")



# BLEU TRANSLATION


def compute_bleu_score(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract the gold (reference) translations and the candidate (whisper) translations
    gold_translations = [sample['gold_translation'] for sample in data]
    whisper_translations = [sample['whisper_translation'] for sample in data]

    #   - a list of reference lists as the second argument (even if you have a single reference per candidate).
    bleu = sacrebleu.corpus_bleu(whisper_translations, [gold_translations])


    print(f"BLEU score: {bleu.score:.2f}")



def main():
    # SETUP

    # get_fleurs_file_codes()
    # get_gold_translation()
    whisper_model = get_whisper("openai/whisper-large-v3")  # Load Whisper model


    # GET LANG DATA
    # french_file_names = get_folder_names(LANG_CODE)
    # for file_path in french_file_names:
    #     get_fleurs_data(file_path, LANG_CODE)


    # TRANSCRIPTION AND TRANSLATION TASK
    # datasets = ["dev", "test", "train"]
    # for dataset in datasets:
    #     transcribe_dataset(ds=dataset, whisper_model = whisper_model)  
    transcribe_dataset(ds="train", whisper_model = whisper_model)  
    output_json = "fr_fr_whisper_out_large.json"
    if os.path.exists(output_json):
        translate_dataset(source_language="fr", target_language="en", ds=output_json)

    gold_codes_matching('fr_fr_whisper_out_large.json', 'fr_fr_codes_file.csv', 'fr_fr_whisper_out_large.json')
    gold_translation_matching('fr_fr_whisper_out_large.json', 'en_translations_file.csv', 'out.json')


    # BLEU
    compute_bleu_score("fr_fr_whisper_out_large.json")
    

    
main()