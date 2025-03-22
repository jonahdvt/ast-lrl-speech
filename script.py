# Installs and imports 

#!pip install transformers torch
#!pip install bs4
#!pip install tf-keras
#!pip install pydub
#!pip install tiktoken
#! pip install soundfile



from config import *
import evaluation
import data
import matching
from indic_seamless import *

from transformers import pipeline, AutoProcessor, SeamlessM4Tv2Model, SeamlessM4Tv2ForSpeechToText, SeamlessM4TTokenizer, SeamlessM4TFeatureExtractor
import torch
import requests
from bs4 import BeautifulSoup
import json
import os
import shutil
import soundfile as sf  





def get_whisper(model='openai/whisper-large-v3', sc_language=None):
    
    if sc_language not in WHISPER_LANGUAGE_CODE_MAPPING:
        raise ValueError(f"Invalid language code: {sc_language}")

    whisper = pipeline(
        "automatic-speech-recognition",
        model=model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device="cuda:0"
    )

    # Return a function that correctly handles `return_timestamps`
    def transcribe(audio_path, return_timestamps=False):
        return whisper(audio_path, return_timestamps=return_timestamps, generate_kwargs={"language": WHISPER_LANGUAGE_CODE_MAPPING[sc_language]})

    return transcribe

# Transcription (ASR)

def get_whisper_transcription(whisper, audio_file):
    transcription = whisper(audio_file, return_timestamps=True)
    return transcription['text']


def transcribe_dataset_whisper(source_language, ds, whisper_model):
    output_dir = f"fleurs_{source_language}_audio/{ds}"  # Path to directory
    output_file = f"{source_language}_aggregate.json"
    num_of_files = (len(os.listdir(output_dir)))
    counter = 0

    results = []  # Store results as a list of dictionaries

    for file_name in os.listdir(output_dir):  # List files in the directory
        if file_name.endswith(".wav"):  # Assuming audio files are .wav (adjust if needed)
            file_path = os.path.join(output_dir, file_name)  # Full path to the file
            transcript = get_whisper_transcription(audio_file=file_path, whisper=whisper_model)  # Pass the full path\
            counter += 1
            print(f"Transcribed {counter}/{num_of_files} files ({round((counter/num_of_files)*100, 2)}%)")
            results.append({
                "file_name": file_name,
                "whisper_transcript": transcript,
            })
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)



def seamless(mode=None, input_dir=None, output_file=None, src_lang=None, tgt_lang="eng", indic=False):

    if mode is None or mode.lower() not in ["transcribe", "translate"]:
        raise ValueError("Invalid mode. Mode must be 'transcribe' or 'translate'.")

    if not indic:
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    else:
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained("ai4bharat/indic-seamless").to("cuda")
        processor = AutoProcessor.from_pretrained("ai4bharat/indic-seamless")

    smls_language_code_mapping = {  # mapping from Fleurs codes to seamless desired inputs
        "hi_in": "hin",
        "pa_in": "pan",
        "ta_in": "tam",
        "te_in": "tel",
        "ml_in": "mal",
        "sw_ke": "swh",
        "ig_ng": "ibo",
        "yo_ng": "yor",
        "lg_ug": "lug",
        "fr": "fra",
        "en": "eng"
    }

    dataset_dir = f"fleurs_{src_lang}_audio/{input_dir}"  # formatting of dataset path

    num_of_files = len(os.listdir(dataset_dir))  # used to track progress
    counter = 0

    audio_files = [  # all audio files (only .wav files)
        os.path.join(dataset_dir, filename)
        for filename in os.listdir(dataset_dir)
        if filename.lower().endswith(('.wav'))
    ]

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as json_file:
            output_data = json.load(json_file)
    else:
        output_data = []

    for audio_file in audio_files:  # iterate over all audio files
        
        if mode.lower() == "transcribe":
            counter += 1
            print(f"Transcribed {counter}/{num_of_files} files ({round((counter/num_of_files)*100, 2)}%)")
            audio_array, sample_rate = sf.read(audio_file)
            audio_sample = {"array": audio_array, "sampling_rate": sample_rate}

            audio_inputs = processor(audios=audio_sample["array"], sampling_rate=16000, return_tensors="pt")
            if indic:
                audio_inputs = {k: v.to("cuda") for k, v in audio_inputs.items()}
            output_tokens = model.generate(
                **audio_inputs,
                tgt_lang=smls_language_code_mapping[src_lang],
            )
            transcribed_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

            if not indic:
                output_data.append({
                    "file_name": os.path.basename(audio_file),
                    "seamless_transcript": transcribed_text_from_audio,
                })
            else:
                output_data.append({
                    "file_name": os.path.basename(audio_file),
                    "seamless_indic_transcript": transcribed_text_from_audio,
                })

        else:  # translate mode
            counter += 1
            print(f"Translated {counter}/{num_of_files} files ({round((counter/num_of_files)*100, 2)}%)")
            
            audio_array, sample_rate = sf.read(audio_file)
            audio_sample = {"array": audio_array, "sampling_rate": sample_rate}
            audio_inputs = processor(audios=audio_sample["array"], sampling_rate=16000, return_tensors="pt")
            if indic:
                audio_inputs = {k: v.to("cuda") for k, v in audio_inputs.items()}
            output_tokens = model.generate(
                **audio_inputs,
                tgt_lang=smls_language_code_mapping[tgt_lang],
            )

            translated_text_from_audio = processor.decode(output_tokens[0], skip_special_tokens=True)
            print(translated_text_from_audio)
            print("Output tokens:", output_tokens)
   
            for sample in output_data:
                if sample["file_name"] == os.path.basename(audio_file):
                    print("check 1 - found the file")
                    if not indic:
                        if "seamless_translation" not in sample:
                            sample["seamless_translation"] = translated_text_from_audio
                            print(f"Added seamless_translation for {os.path.basename(audio_file)}")
                        else:
                            print(f"seamless_translation already exists for {os.path.basename(audio_file)}")
                    else:
                        if "seamless_indic_translation" not in sample:
                            sample["seamless_indic_translation"] = translated_text_from_audio
                            print(f"Added seamless_indic_translation for {os.path.basename(audio_file)}")
                        else:
                            print(f"seamless_indic_translation already exists for {os.path.basename(audio_file)}")
                else:
                    if not indic:
                        output_data.append({
                            "file_name": os.path.basename(audio_file),
                            "seamless_translation": translated_text_from_audio,
                        })
                        print(f"Appended seamless_translation for {os.path.basename(audio_file)}")
                    else:
                        output_data.append({
                            "file_name": os.path.basename(audio_file),
                            "seamless_indic_translation": translated_text_from_audio,
                        })
                    print(f"Appended seamless_indic_translation for {os.path.basename(audio_file)}")
    
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)



# Translations

def translate_dataset_nllb(source_language=None, target_language="en", json_ds=None):

    NLLB_LANGUAGE_CODE_MAPPING = {
        "hi_in": "hin_Deva",
        "pa_in": "pan_Guru",
        "ta_in": "tam_Taml",
        "te_in": "tel_Telu",
        "ml_in": "mal_Mlym",
        "sw_ke": "swh_Latn",
        "ha_ng": "hau_Latn",
        "ig_ng": "ibo_Latn",
        "yo_ng": "yor_Latn",
        "lg_ug": "lug_Latn",
        "fr_fr": "fra_Latn",
        "en": "eng_Latn"
    }

    if source_language not in NLLB_LANGUAGE_CODE_MAPPING:
        raise ValueError(f"Source language {source_language} not supported for NLLB.")
    if target_language not in NLLB_LANGUAGE_CODE_MAPPING:
        raise ValueError(f"Target language {target_language} not supported for NLLB.")

    # Create the translation pipeline using the NLLB distilled model.
    translator = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-1.3B",
        src_lang=NLLB_LANGUAGE_CODE_MAPPING[source_language],
        tgt_lang=NLLB_LANGUAGE_CODE_MAPPING[target_language]
    )

    with open(json_ds, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_entries = len(data)
    counter = 0

    for sample in data:

        if (("seamless_transcript" in sample or "whisper_transcript" in sample) and "nllb_translation" not in sample):
            
            source_text = sample.get("seamless_transcript", sample.get("whisper_transcription"))
            translation = translator(source_text)
            sample["nllb_translation"] = translation[0]['translation_text']

            counter += 1
            print(f"NLLB translated {counter}/{num_entries} samples.")
        

    # Save the updated dataset back to the JSON file.
    with open(json_ds, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_gt_translation(text, source_language, target_language):
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


def translate_dataset_gt(source_language, target_language="en", ds=None):
    with open(ds, 'r', encoding="utf-8") as f:
        data = json.load(f)
    num_entries = len(data)
    counter = 0
    for sample in data:
        if "transcript" in sample and "gt_translation" not in sample:  # Avoid redundant translation
            sample["gt_translation"] = get_gt_translation(sample["transcript"], source_language, target_language)
            print(f"Translated {counter}/{num_entries} of the entries")
            counter += 1

    with open(ds, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)







def main():
    for sc_lang_code in FLEURS_LANGUAGE_CODES:
    #     evaluation.compute_wer(sc_lang_code)
    #     sc_language_file_names = data.get_folder_names(sc_lang_code)                      # get all set names from fleurs for source language
    #     for file_path in sc_language_file_names:
    #         data.get_fleurs_data(file_path, sc_lang_code)       

         aggregate_json = f"lang_aggregate_data/test/{sc_lang_code}_aggregate.json"
         indic_seamless(sc_lang_code, aggregate_json)  

    #     datasets = ["test"]                     
    #     for ds in datasets:
    #         seamless(mode="transcribe",input_dir=ds, output_file=aggregate_json, src_lang=sc_lang_code, tgt_lang=sc_lang_code)  
    #         seamless(mode="translate",input_dir=ds, output_file=aggregate_json, src_lang=sc_lang_code, tgt_lang="en", indic=True)  
    #     translate_dataset_nllb(sc_lang_code, "en", aggregate_json)
         print(f"finished translation of {sc_lang_code}")

         matching.gold_codes_matching(aggregate_json, f'fleurs_lang_info/{sc_lang_code}_fleurs_info.csv', aggregate_json)
         matching.gold_translation_matching(aggregate_json, 'fleurs_lang_info/en_translations.csv', aggregate_json)

         evaluation.compute_bleu_score(aggregate_json, sc_lang_code, mode="seamless_indic")

    #     shutil.rmtree(f"fleurs_{sc_lang_code}_audio")
    #     print(f"Deleted folder: fleurs_{sc_lang_code}_audio")
    # sc_lang_code = "fr_fr"
    # evaluation.compute_wer(sc_lang_code)
    # aggregate_json = f"lang_aggregate_data/{sc_lang_code}_aggregate.json"
    # translate_dataset_nllb(sc_lang_code, "en", aggregate_json)
    # evaluation.compute_bleu_score(aggregate_json, sc_lang_code)
    




    
if __name__ == "__main__":
    main()
