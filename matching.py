import json
import pandas as pd

def gold_codes_matching(json_path, csv_path, output_path):
    """
    Matches transcript entries in a JSON file with corresponding codes from a CSV file.

    The CSV is expected to have columns 'wav_codes' and 'codes'. The JSON file should contain
    entries with a 'file_name' key. This function maps each 'file_name' in the JSON to a code
    from the CSV and adds a new key 'code' to the JSON entry.

    Parameters:
        json_path (str): Path to the input JSON file.
        csv_path (str): Path to the CSV file containing the codes.
        output_path (str): Path where the updated JSON will be saved.
    """
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Load CSV data
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    # Create a mapping from wav_codes to codes
    name_mapping = dict(zip(df['wav_codes'], df['codes']))

    # Update JSON entries with the matching code
    for entry in json_data:
        file_name = entry.get('file_name')
        if file_name in name_mapping:
            entry['code'] = name_mapping[file_name]

    # Save the updated JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)



def gold_text_matching(json_path, csv_path, output_path, mode): # either put same language for transcript, or english from translation as the csv
    """
    Matches the audio code in a JSON file with the corresponding gold translation
    from a CSV file.

    The CSV should have columns 'codes' and 'transcript'. Each JSON entry is expected to have a 'code'
    key which is used to match with the CSV. This function adds a new key 'gold_translation'
    to each JSON entry with the corresponding transcript.

    Parameters:
        json_path (str): Path to the input JSON file.
        csv_path (str): Path to the CSV file containing the gold translations.
        output_path (str): Path where the updated JSON will be saved.
    """
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Create a mapping from codes to transcripts
    name_mapping = dict(zip(df['codes'], df['transcript']))

    # Update JSON entries with the gold translation
    for entry in json_data:
        file_code = entry.get('code')
        if file_code in name_mapping and mode.lower() == "translation":
            entry['gold_translation'] = name_mapping[file_code]
        elif file_code in name_mapping and mode.lower() == "transcription":
            entry['gold_transcript'] = name_mapping[file_code]

    # Save the updated JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"Updated JSON saved to {output_path}")


