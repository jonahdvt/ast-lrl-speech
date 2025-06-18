from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline,WhisperFeatureExtractor,WhisperTokenizer
import evaluate
# Assuming config.py contains NLLB_LANGUAGE_CODE_MAPPING and WHISPER_LANGUAGE_CODE_MAPPING
from mapping import NLLB_LANGUAGE_CODE_MAPPING, WHISPER_LANGUAGE_CODE_MAPPING 
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

language_codes = [
    "hi_in",
    "pa_in", 
    "ta_in", 
    "te_in", 
    "ml_in",
    "sw_ke",
    "ha_ng",
    "yo_ng",
    "lg_ug",
    "ig_ng",
    # "fr_fr",
]

# model_lang = [
#     # "hi_in",
#     # "pa_in", 
#     # "ta_in", 
#     # "te_in", 
#     "ml_in",
#     # "sw_ke",
#     # "ha_ng",
#     # "yo_ng",
#     # "lg_ug",
#     # "ig_ng",
# ]

# test_lang = [
#     "hi_in",
#     "pa_in", 
#     "ta_in", 
#     "te_in", 
#     "ml_in",
#     # "sw_ke",
#     # "ha_ng",
#     # "yo_ng",
#     # "lg_ug",
#     # "ig_ng",
#     # "fr_fr",
# ]


def translate_dataset_nllb(source_language: str, target_language: str, json_ds: str):
    """
    Translate text entries in a JSON dataset file using the NLLB translation pipeline.

    Args:
        source_language (str): Source language code in our mapping.
        target_language (str): Target language code (defaults to 'en').
        json_ds (str): Path to the JSON dataset file to update.
    """
    logging.info(f"Starting NLLB translation for {source_language} to {target_language} in {json_ds}")

    if source_language not in NLLB_LANGUAGE_CODE_MAPPING:
        logging.error(f"Source language {source_language} not supported for NLLB.")
        raise ValueError(f"Source language {source_language} not supported for NLLB.")
    if target_language not in NLLB_LANGUAGE_CODE_MAPPING:
        logging.error(f"Target language {target_language} not supported for NLLB.")
        raise ValueError(f"Target language {target_language} not supported for NLLB.")

    try:
        translator = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-1.3B",
            src_lang=NLLB_LANGUAGE_CODE_MAPPING[source_language],
            tgt_lang=NLLB_LANGUAGE_CODE_MAPPING[target_language],
            max_length=400,
        )
        logging.info("NLLB translation pipeline initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize NLLB translation pipeline: {e}")
        raise

    try:
        with open(json_ds, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded data from {json_ds}")
    except FileNotFoundError:
        logging.error(f"JSON dataset file not found: {json_ds}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {json_ds}: {e}")
        return

    for i, sample in enumerate(data):
        if "asr_corrector_pred" in sample and "nllb_translation_asr" not in sample:
            source_text = sample["asr_corrector_pred"]
            try:
                translation = translator(source_text)[0]['translation_text']
                sample["nllb_translation_asr"] = translation
                # logging.debug(f"Translated sample {i}: '{source_text}' -> '{translation}'")
            except Exception as e:
                logging.warning(f"Error translating sample {i} ('{source_text}'): {e}")
                sample["nllb_translation_asr"] = "" # Assign empty string or handle as appropriate
        else:
            logging.debug(f"Skipping sample {i}: 'asr_corrector_pred' missing or 'nllb_translation_asr' already exists.")

    try:
        with open(json_ds, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully wrote translated data back to {json_ds}")
    except IOError as e:
        logging.error(f"Error writing translated data to {json_ds}: {e}")


def test_asr_model(language_code_a: str, language_code_b: str, results_dir: str = "MASSIVE_PLUS") -> str:
    """
    Run ASR test for a given language on the FLEURS test dataset, compute WER,
    save raw transcriptions to a JSON file, and return the file path.

    Args:
        language_code_a (str): Language code of the model
        language_code_b (str): Language code to test.
        results_dir (str): Directory to store result JSON files.

    Returns:
        str: Path to the JSON file containing transcription results.
    """
    logging.info(f"Starting ASR test: Model language '{language_code_a}', Test language '{language_code_b}'")
    model_name = f"jonahdvt/whisper-fleurs-large-plus-{language_code_a}"

    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        tokenizer = WhisperTokenizer.from_pretrained(
            model_name,
            language=WHISPER_LANGUAGE_CODE_MAPPING[language_code_b],
            task="transcribe"
        )
        processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        logging.info(f"Loaded processor for model: {model_name}")
    except Exception as e:
        logging.error(f"Failed to load Whisper processor components for {model_name}: {e}")
        raise

    # Load FLEURS test split
    try:
        fleurs_test = load_dataset(
            "google/fleurs",
            language_code_b,
            split="test",
            trust_remote_code=True,
            streaming=True
        ).cast_column("audio", Audio(sampling_rate=16000))
        logging.info(f"Loaded FLEURS dataset for language: {language_code_b}")
    except Exception as e:
        logging.error(f"Failed to load FLEURS dataset for {language_code_b}: {e}")
        raise

    # Create ASR pipeline
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            return_timestamps=True
        )
        logging.info("ASR pipeline initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize ASR pipeline for {model_name}: {e}")
        raise

    # wer_metric = evaluate.load("wer") # Not used in this version
    predictions, references, result_list = [], [], []

    output_subdir = os.path.join(results_dir, language_code_a)
    os.makedirs(output_subdir, exist_ok=True)
    json_filename = os.path.join(output_subdir, f"{language_code_b}.json")
    logging.info(f"Output JSON file: {json_filename}")

    # Clear or create JSON file
    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump([], f)
        logging.info(f"Initialized empty JSON file: {json_filename}")
    except IOError as e:
        logging.error(f"Error initializing JSON file {json_filename}: {e}")
        raise

    # Iterate over test samples
    for sample_idx, sample in enumerate(fleurs_test, 1):
        try:
            result = asr_pipeline(sample["audio"]["array"])
            predictions.append(result["text"])
            references.append(sample["transcription"])

            file_name = sample["audio"].get("path", f"sample_{sample_idx}")
            result_list.append({
                "file_name": file_name,
                "whisper_l_ft": result["text"]
            })
            if sample_idx % 100 == 0:
                logging.info(f"Processed {sample_idx} samples for {language_code_b} on model {language_code_a}.")
        except Exception as e:
            logging.error(f"Error processing sample {sample_idx} for {language_code_b}: {e}")
            # Depending on desired behavior, you might want to continue or skip this sample
            continue

    # wer_value = wer_metric.compute(predictions=predictions, references=references)
    # print(f"Final WER for {language_code}: {wer_value:.3f}")
    logging.info(f"Finished testing {language_code_b} on model {model_name}. Total samples processed: {len(result_list)}")

    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(result_list, f, indent=2, ensure_ascii=False)
        logging.info(f"Transcription results saved to {json_filename}")
    except IOError as e:
        logging.error(f"Error saving transcription results to {json_filename}: {e}")
        raise

    return json_filename


if __name__ == "__main__":
    # Test cross-language ASR models
    # logging.info("Starting cross-language ASR testing loop.")
    # for m_lang in model_lang: # Using pre-defined model_lang list
    #     for t_lang in test_lang: # Using pre-defined test_lang list
    #         if m_lang != t_lang:
    #             logging.info(f"Initiating ASR test: Model {m_lang} vs Test {t_lang}")
    #             try:
    #                 ds_path = test_asr_model(m_lang, t_lang)
    #                 logging.info(f"Completed ASR test: Model {m_lang} on {t_lang}. Results saved in {ds_path}")
    #             except Exception as e:
    #                 logging.critical(f"Failed ASR test for Model {m_lang} on {t_lang}: {e}", exc_info=True)
    #         else:
    #             logging.info(f"Skipping ASR test for {m_lang} on {t_lang} (model and test language are the same).")

    # Example of how you would use the translation function if needed:
    # results_dirs = [
    #     # "MASSIVE_PLUS", 
    #     "asr_corrector/prediction_plus_json"
    #     ] 
        # This should match the directory used by the ASR test loop
    results_dir = "asr_corrector/prediction_plus_json"
    logging.info("\nStarting NLLB translation loop for all expected ASR files.")
    # for results_dir in results_dirs:
    #     for m_lang in model_lang:
    for t_lang in language_codes:
                # Construct the expected path for the ASR output JSON file
        # output_subdir = os.path.join(results_dir, m_lang)
        json_filename = os.path.join(results_dir, f"{t_lang}.json")

        if os.path.exists(json_filename):
            try:
                        # The source language for translation is the language of the ASR test
                translate_dataset_nllb(
                    source_language=t_lang,
                    target_language="en", # Translating to English
                    json_ds=json_filename
                )
                logging.info(f"Completed NLLB translation for {t_lang} in file {json_filename}")
            except Exception as e:
                logging.critical(f"Failed NLLB translation for {t_lang} from {json_filename}: {e}", exc_info=True)
        else:
            logging.warning(f"Skipping translation for {t_lang}: ASR result file not found at {json_filename}. Ensure ASR tests were run first.")

    # logging.info("All NLLB translations completed.")
    # Specific example for a single language translation
    # specific_lang_to_translate = "yo_ng"
    # specific_ds_path = f"MASSIVE/yo_ng/yo_ng.json" # Assuming this file exists from a previous ASR run
    # if os.path.exists(specific_ds_path):
    #     logging.info(f"Attempting NLLB translation for {specific_lang_to_translate} using file: {specific_ds_path}")
    #     try:
    #         translate_dataset_nllb(source_language=specific_lang_to_translate, target_language="en", json_ds=specific_ds_path)
    #         logging.info(f"Successfully completed NLLB translation for {specific_lang_to_translate}.")
    #     except Exception as e:
    #         logging.critical(f"Failed NLLB translation for {specific_lang_to_translate}: {e}", exc_info=True)
    # else:
    #     logging.warning(f"Skipping translation for {specific_lang_to_translate}: File not found at {specific_ds_path}")

    logging.info("Script execution finished.")