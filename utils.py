from config import FLEURS_LANGUAGE_CODES
import json

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


rename_json_property("lang_aggregate_data/fr_fr_aggregate.json", "whisper_transcription", "whisper_transcript")