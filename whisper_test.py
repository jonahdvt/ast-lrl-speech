from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline,WhisperFeatureExtractor,WhisperTokenizer
import evaluate
from config import *
import json
import os 

language_codes = [
    "hi_in",
    "pa_in", 
    "ta_in", 
    "te_in", 
    "ml_in"
    ]


def translate_dataset_nllb(source_language=None, target_language="en", json_ds=None):

    if source_language not in NLLB_LANGUAGE_CODE_MAPPING:
        raise ValueError(f"Source language {source_language} not supported for NLLB.")
    if target_language not in NLLB_LANGUAGE_CODE_MAPPING:
        raise ValueError(f"Target language {target_language} not supported for NLLB.")

    # Create the translation pipeline using the NLLB distilled model.
    translator = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-1.3B",
        src_lang=NLLB_LANGUAGE_CODE_MAPPING[source_language],
        tgt_lang=NLLB_LANGUAGE_CODE_MAPPING[target_language],
        max_length=400,
    )

    with open(json_ds, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_entries = len(data)
    counter = 0

    for sample in data:

        if (("whisper_s_ft" in sample) and "nllb_translation" not in sample):

            source_text = sample.get("whisper_s_ft")
            translation = translator(source_text)
            sample["nllb_translation"] = translation[0]['translation_text']

            counter += 1


    # Save the updated dataset back to the JSON file.
    with open(json_ds, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



# 1. Setup for-loop

for language_code in language_codes:
# Path to the directory where the model & processor were saved (or the Hub identifier)
    model_dir = f"jonahdvt/whisper-fleurs-small-indic"
    # model_dir = f"jonahdvt/whisper-fleurs-small-{language_code}"

# 2. Load the FLEURS test dataset
    fleurs_test = load_dataset("google/fleurs", language_code, split="test", trust_remote_code=True)
    fleurs_test = fleurs_test.cast_column("audio", Audio(sampling_rate=16000))

# 3. Load the trained model and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_dir,
        language=WHISPER_LANGUAGE_CODE_MAPPING[language_code],
        task="transcribe"
    )
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)


# 3b. Load the classic model and processor
    #model = 'openai/whisper-small'
    #feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
    #tokenizer = WhisperTokenizer.from_pretrained(model, language=WHISPER_LANGUAGE_CODE_MAPPING[language_code], task="transcribe")
    #processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# 4. (Optional) Create a Hugging Face pipeline for ASR
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        return_timestamps=True
    )

# 5. Evaluate the model on the test set
    wer_metric = evaluate.load("wer")
    predictions = []
    result_list = [] # to store the was codes and transcriptions
    references = []
    count = 0

# Create the directory if it doesn't exist
    results_dir = "s_ft_whisper_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

# Set JSON filename based on the language code
    json_filename = f"{results_dir}/{language_code}_indic.json"

    if os.path.exists(json_filename):
        with open(json_filename, "w") as f:
            json.dump([], f)



# 6 Iterate over test samples (this example uses a simple loop; for large datasets consider batching)
    for sample in fleurs_test:
        count+=1
    # Use the pipeline to obtain transcription
        result = asr_pipeline(sample["audio"]["array"])

        predictions.append(result["text"])
        references.append(sample["transcription"])


    
    # Use the audio file's path as its unique code (fallback to sample index if not available)
        wav_code = sample["audio"].get("path", f"sample_{count}")
        result_list.append({
            "wav_code": wav_code,
            "whisper_s_ft": result["text"]
        })
        wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"Final WER of {language_code} on FLEURS test set:", wer)

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)





    translate_dataset_nllb(source_language=language_code, target_language="en", json_ds=json_filename)


