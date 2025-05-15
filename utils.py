from config import FLEURS_LANGUAGE_CODES
import json
import matching
import evaluation 
from typing import Tuple

def rename_json_property(file=None, old_name=None, new_name=None):
    with open(file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    # If the JSON is a list of dictionaries, iterate over each element.
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and old_name in item:
                item[new_name] = item.pop(old_name)
    # If it's just a single dictionary, handle it directly.
    elif isinstance(data, dict):
        if old_name in data:
            data[new_name] = data.pop(old_name)
        else:
            raise KeyError(f"Key '{old_name}' not found in the JSON data.")
    else:
        raise TypeError("Unexpected JSON structure. Expected a list or a dictionary.")
    
    # Write the updated data back to the file
    with open(file, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Renamed for file : {file}")




def remove_prefix_from_wav_code(input_file, output_file, prefix="test/"):
    """
    Loads JSON data from input_file, removes the specified prefix from each 'wav_code' entry,
    and writes the updated data to output_file.

    Parameters:
        input_file (str): Path to the JSON file containing the data.
        output_file (str): Path where the updated JSON will be saved.
        prefix (str): The prefix to remove from 'wav_code' entries (default is "test/").
    """
    # Load JSON data from the file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Iterate through each entry and remove the prefix if it exists
    for entry in data:
        if entry.get("file_name", "").startswith(prefix):
            entry["file_name"] = entry["file_name"][len(prefix):]

    # Write the updated data to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def average_words_per_entry(json_path: str) -> Tuple[int, int, float]:
    total_words = 0
    total_entries = 0

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            transcript = entry.get("gold_transcript", "").strip()
            if transcript:
                total_entries += 1
                total_words += len(transcript.split())

    average = (total_words / total_entries) if total_entries else 0.0
    print(json_path, average)
    return total_entries, total_words, average



codes = [
    # "hi_in",
#     # "pa_in", 
#     # "ta_in", 
    # "te_in", 
#     #"ml_in",
    # "sw_ke",
    # "ha_ng",
    # "yo_ng",
    # "ig_ng",
    # "lg_ug",
#     ]
# codes = [
    # "1h",
    "5h", 
    # "10h", 
    # "20h"
    ]
for code in codes:

    # file = f"whisper_l_train_data/{code}.json"
    file = f"whisper_l_hours_results/ig_ng/{code}.json"
    lang = "ig_ng"

    remove_prefix_from_wav_code(file, file)
    rename_json_property(file, "file_id", "file_name")
    matching.gold_codes_matching(file, f"fleurs_lang_info/{code}_fleurs_info.csv", file)
    matching.gold_text_matching(file, "fleurs_lang_info/en_translations.csv", file, "translation") # Translations
    evaluation.compute_bleu_score(file, code, "nllb")
    

    # matching.gold_text_matching(file, f"fleurs_lang_info/{code}_fleurs_info.csv", file, "transcription") # Transcripts
    # print(evaluation.detailed_wer(file))
    # average_words_per_entry(file)











