from config import FLEURS_LANGUAGE_CODES
import json
import matching
import evaluation 

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


code = "sw_ke"
file = f"s_ft_whisper_results/{code}_afri.json"
rename_json_property(file, "wav_code", "file_name")
remove_prefix_from_wav_code(file, file)
matching.gold_codes_matching(file, f"fleurs_lang_info/{code}_fleurs_info.csv", file)
matching.gold_translation_matching(file, "fleurs_lang_info/en_translations.csv", file)
evaluation.compute_bleu_score(file, code, "nllb")










