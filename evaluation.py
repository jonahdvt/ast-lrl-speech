import json
import sacrebleu
from jiwer import wer, Compose, RemoveEmptyStrings, ToLowerCase, RemoveMultipleSpaces, Strip, RemovePunctuation, ReduceToListOfListOfWords
import pandas as pd
import re

def compute_wer(sc_lang):
    """
    Computes the average Word Error Rate (WER) between whisper transcriptions
    (from the JSON file) and the reference transcripts (from the CSV file) by
    matching records on "file_name" and "wav_codes".

    The function standardises both inputs using a jiwer transformation pipeline:
      - Removes empty strings
      - Converts to lowercase
      - Removes multiple spaces
      - Strips leading/trailing whitespace
      - Removes punctuation
      - Reduces the string to a list of lists of words

    Parameters:
        sc_lang (str): The source language code used in the filenames.

    Returns:
        avg_wer (float): The average WER score computed over all matching files.
                         Returns None if no matching files are found.
    """
    json_file = f"lang_aggregate_data/{sc_lang}_aggregate.json"
    csv_file = f"fleurs_lang_info/{sc_lang}_fleurs_info.csv"
    
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    df = pd.read_csv(csv_file, encoding="utf-8")
    
    transcript_mapping = dict(zip(df["wav_codes"], df["transcript"]))
    
    total_wer = 0.0
    count = 0
    
    # transformation pipeline for standardizing inputs
    transforms = Compose([
        RemoveEmptyStrings(),
        ToLowerCase(),
        RemoveMultipleSpaces(),
        Strip(),
        RemovePunctuation(),
        ReduceToListOfListOfWords()
    ])
    
    for record in json_data:
        file_name = record.get("file_name")
        whisper_transcription = record.get("whisper_transcript")
        
        if file_name in transcript_mapping and whisper_transcription is not None:
            reference_transcript = transcript_mapping[file_name]
            score = wer(
                reference_transcript,
                whisper_transcription,
                truth_transform=transforms,
                hypothesis_transform=transforms
            )
            total_wer += score
            count += 1

        
    if count > 0:
        avg_wer = total_wer / count
        print(f"Average WER: {avg_wer:.3f} for {sc_lang}")
        return avg_wer
    else:
        print("No matching records found.")
        return None





def detokenize(text):
    """
    A simple detokenizer that removes spaces before punctuation.
    This may be adjusted for language-specific needs.
    """
    return re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)



def compute_bleu_score(file, language_code, mode, force=False):
    """
    Computes the BLEU score using sacreBLEU for the provided translations.
    
    Parameters:
      file (str): Path to the JSON file containing translation samples.
      language_code (str): The language code for reporting.
      mode (str): Which translation mode to evaluate ('seamless', 'seamless_indic', or 'nllb').
      force (bool): If True, assumes the data is already detokenized and skips detokenization.
    """
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize lists for gold translations and hypothesis translations
    gold_translations = []
    seamless_hypothesis_translations = []
    nllb_hypothesis_translations = []
    seamless_indic_hypothesis_translations = []

    for sample in data:
        # Check if 'gold_translation' exists in the sample
        if 'gold_translation' in sample:
            gold_text = sample['gold_translation']
            if not force:
                gold_text = detokenize(gold_text)
            gold_translations.append(gold_text)
        else:
            print(f"Warning: Missing gold translation in file {sample.get('file_name', 'Unknown file')}")
            continue

        # Process based on the chosen mode, applying detokenization if required
        if mode.lower() == "seamless" and 'seamless_translation' in sample:
            translation = sample['seamless_translation']
            if not force:
                translation = detokenize(translation)
            seamless_hypothesis_translations.append(translation)
        
        elif mode.lower() == "seamless_indic" and 'translation' in sample:
            translation = sample['translation']
            if not force:
                translation = detokenize(translation)
            seamless_indic_hypothesis_translations.append(translation)
        
        elif mode.lower() == "nllb" and 'nllb_translation' in sample:
            translation = sample['nllb_translation']
            if not force:
                translation = detokenize(translation)
            nllb_hypothesis_translations.append(translation)

    # Calculate BLEU score for seamless translations (if available)
    if seamless_hypothesis_translations:
        bleu_seamless = sacrebleu.corpus_bleu(seamless_hypothesis_translations, [gold_translations])
        print(f"{language_code}, seamless, bleu = {bleu_seamless.score:.2f}")

    # Calculate BLEU score for seamless indic translations (if available)
    if seamless_indic_hypothesis_translations:
        bleu_seamless_indic = sacrebleu.corpus_bleu(seamless_indic_hypothesis_translations, [gold_translations])
        print(f"{language_code}, indic seamless, bleu = {bleu_seamless_indic.score:.2f}")

    # Calculate BLEU score for nllb translations (if available)
    if nllb_hypothesis_translations:
        bleu_nllb = sacrebleu.corpus_bleu(nllb_hypothesis_translations, [gold_translations])
        print(f"{language_code}, nllb, bleu = {bleu_nllb.score:.2f}")


    # In case both seamless and nllb translations are available, handle both
    if seamless_hypothesis_translations and nllb_hypothesis_translations:
        print(f"{language_code}, both seamless and nllb available.")

