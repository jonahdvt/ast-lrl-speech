# Installs and imports 

#!pip install transformers torch
#!pip install bs4
#!pip install tf-keras
#!pip install pydub


from transformers import pipeline
import torch
import sacrebleu


import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import tarfile
import os
import json
import shutil




base_url = "https://huggingface.co/datasets/google/fleurs-r/resolve/main/"


language_codes = [
    # "hi_in",  # Hindi     DONE
    "pa_in",  # Punjabi   dev+test are transcribed 
    # "ta_in",  # Tamil     DONE 
    # "te_in",  # Telugu
    # "ml_in",  # Malayalam
    # "sw_ke",  # Swahili
    # "ha_ng",  # Hausa
    # "ig_ng",  # Igbo
    "yo_ng",  # Yoruba
    # "lg_ug"   # Luganda
]




def get_folder_names(language_code=None):
    return [
        f"data/{language_code}/audio/dev.tar.gz",
        f"data/{language_code}/audio/test.tar.gz",
        f"data/{language_code}/audio/train.tar.gz"
    ]

def get_fleurs_data(file_path, language_code=None):  
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


def get_fleurs_file_codes(language_code=None):
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





def get_whisper(model='openai/whisper-large-v3', sc_language=None):
    
    whisper_language_code_mapping = {
        "hi_in": "Hindi",  
        "pa_in": "Punjabi",  
        "ta_in": "Tamil",  
        "te_in": "Telugu", 
        "ml_in": "Malayalam",  
        "sw_ke": "Swahili",  
        "ha_ng": "Hausa",   
        "ig_ng": "Igbo",  
        "yo_ng": "Yoruba",  
        "lg_ug": "Luganda" 
    }

    if sc_language not in whisper_language_code_mapping:
        raise ValueError(f"Invalid language code: {sc_language}")

    whisper = pipeline(
        "automatic-speech-recognition",
        model=model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device="cuda:0"
    )

    # Return a function that correctly handles `return_timestamps`
    def transcribe(audio_path, return_timestamps=False):
        return whisper(audio_path, return_timestamps=return_timestamps, generate_kwargs={"language": whisper_language_code_mapping[sc_language]})

    return transcribe




def get_transcription(whisper, audio_file):
    transcription = whisper(audio_file, return_timestamps=True)
    return transcription['text']


def transcribe_dataset(source_language, ds, whisper_model):
    output_dir = f"fleurs_{source_language}_audio/{ds}"  # Path to directory
    output_file = f"{source_language}_aggregate.json"
    num_of_files = (len(os.listdir(output_dir)))
    counter = 0

    results = []  # Store results as a list of dictionaries

    for file_name in os.listdir(output_dir):  # List files in the directory
        if file_name.endswith(".wav"):  # Assuming audio files are .wav (adjust if needed)
            file_path = os.path.join(output_dir, file_name)  # Full path to the file
            transcript = get_transcription(audio_file=file_path, whisper=whisper_model)  # Pass the full path\
            counter += 1
            print(f"Transcribed {counter}/{num_of_files} files ({round((counter/num_of_files)*100, 2)}%)")
            results.append({
                "file_name": file_name,
                "transcript": transcript,
            })
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)





def get_translation(text, source_language, target_language):
    # Get the translation from the sample text
    gt_language_code_mapping = {
        "hi_in": "hi",  
        "pa_in": "pa",  
        "ta_in": "ta",  
        "te_in": "te", 
        "ml_in": "ml",  
        "sw_ke": "sw",  
        "ha_ng": "ha",   
        "ig_ng": "ig",  
        "yo_ng": "yo",  
        "lg_ug": "lg" 
    }
    
    url = f"https://translate.google.com/m?tl={target_language}&sl={gt_language_code_mapping[source_language]}&q={text}"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}")

    # Parse the page content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the translated text from the page
    translation = soup.find('div', class_='result-container').text

    return translation

def translate_dataset(source_language, target_language="en", ds=None):
    with open(ds, 'r', encoding="utf-8") as f:
        data = json.load(f)
    num_entries = len(data)
    counter = 0
    for sample in data:
        if "transcript" in sample and "whisper_translation" not in sample:  # Avoid redundant translation
            sample["whisper_translation"] = get_translation(sample["transcript"], source_language, target_language)
            print(f"Translated {counter}/{num_entries} of the entries")
            counter += 1

    with open(ds, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)





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

def compute_bleu_score(file, language_code):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize lists for gold and whisper translations
    gold_translations = []
    whisper_translations = []

    for sample in data:
        # Check if 'gold_translation' and 'whisper_translation' exist in the sample
        if 'gold_translation' in sample and 'whisper_translation' in sample:
            gold_translations.append(sample['gold_translation'])
            whisper_translations.append(sample['whisper_translation'])
        else:
            print(f"Warning: Missing translations in file {sample.get('file_name', 'Unknown file')}")
            continue

    # If there are no valid translations, we can't compute BLEU
    if not gold_translations or not whisper_translations:
        print("Error: No valid translations found.")
        return

    bleu = sacrebleu.corpus_bleu(whisper_translations, [gold_translations])
    print(f"{language_code}, whisper/google_translate, bleu = {bleu.score:.2f}\n")

    with open("bleu_score.txt", "a", encoding="utf-8") as f:
        f.write(f"{language_code}, whisper/google_translate, bleu = {bleu.score:.2f}\n")





def main():
    # SETUP
    # get_gold_translation()                                                  # get english transcripts from Fleurs (only done once)


    for sc_lang_code in language_codes:
        get_fleurs_file_codes(language_code=sc_lang_code)                            # get matching audio codes between languages 
        whisper_model = get_whisper("openai/whisper-large-v3", sc_language=sc_lang_code)                    # Load Whisper model
        
        
        # GET LANG DATA
        sc_language_file_names = get_folder_names(sc_lang_code)                      # get all set names from fleurs for source language
        for file_path in sc_language_file_names:
            get_fleurs_data(file_path, sc_lang_code)                                 # fetch data and download audio tar files 


        # TRANSCRIPTION AND TRANSLATION TASK
        datasets = ["dev", "test", "train"]                     
        for dataset in datasets:
            transcribe_dataset(source_language=sc_lang_code, ds=dataset, whisper_model = whisper_model)         # transcribe all audio files to text in source language
        


        aggregate_json = f"{sc_lang_code}_aggregate.json"                    # json file for all operations - translate, append gold translation, compute BLEU
        # if os.path.exists(aggregate_json):                                                                    # Commented for ASR only
        #     translate_dataset(source_language=sc_lang_code, target_language="en", ds=aggregate_json)

        gold_codes_matching(aggregate_json, f'{sc_lang_code}_codes_file.csv', aggregate_json)
        gold_translation_matching(aggregate_json, 'en_translations_file.csv', aggregate_json)


        # BLEU
        # compute_bleu_score(file=aggregate_json, language_code=sc_lang_code)

        shutil.rmtree(f"fleurs_{sc_lang_code}_audio")
        print(f"Deleted folder: fleurs_{sc_lang_code}_audio")
    

    
# if __name__ == "__main__":
#     main()



# Notes 
# Redo translation + BLEU for Hindi, Tamil, Punjabi, Hausa




directory_path = 'lang_aggregate_data'
for filename in os.listdir(directory_path):
    # Check if the file ends with aggregate.json
    if filename.endswith('aggregate.json'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        for sample in data:
            if 'whisper_translation' in sample:
                del sample['whisper_translation']
        
        # Save the modified data back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

print("Whisper translation section deleted from all aggregate.json files.")
