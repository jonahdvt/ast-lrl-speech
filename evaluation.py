import json
import sacrebleu
from jiwer import wer, Compose, RemoveEmptyStrings, ToLowerCase, RemoveMultipleSpaces, Strip, RemovePunctuation, ReduceToListOfListOfWords
import pandas as pd


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
    csv_file = f"gold_transcripts/{sc_lang}_fleurs_transcript_file.csv"
    
    # Load the JSON file
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, encoding="utf-8")
    
    # Build a mapping from wav_codes to their reference transcript
    transcript_mapping = dict(zip(df["wav_codes"], df["transcript"]))
    
    total_wer = 0.0
    count = 0
    
    # Create transformation pipeline for standardizing inputs
    transforms = Compose([
        RemoveEmptyStrings(),
        ToLowerCase(),
        RemoveMultipleSpaces(),
        Strip(),
        RemovePunctuation(),
        ReduceToListOfListOfWords()
    ])
    
    # Iterate over each record in the JSON file
    for record in json_data:
        file_name = record.get("file_name")
        whisper_transcription = record.get("seamless_transcript")
        
        # Ensure both transcription and reference exist
        if file_name in transcript_mapping and whisper_transcription is not None:
            reference_transcript = transcript_mapping[file_name]
            # Compute the WER score using the standardized inputs
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







def compute_bleu_score(file, language_code):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize lists for gold translations and hypothesis translations
    gold_translations = []
    seamless_hypothesis_translations = []
    nllb_hypothesis_translations = []

    for sample in data:
        # Check if 'gold_translation' exists in the sample
        if 'gold_translation' in sample:
            gold_translations.append(sample['gold_translation'])
        else:
            print(f"Warning: Missing gold translation in file {sample.get('file_name', 'Unknown file')}")
            continue

        # Check if 'seamless_translation' exists and append
        if 'seamless_translation' in sample:
            seamless_hypothesis_translations.append(sample['seamless_translation'])

        # Check if 'nllb_translation' exists and append
        if 'nllb_translation' in sample:
            nllb_hypothesis_translations.append(sample['nllb_translation'])

    # Calculate BLEU score for seamless translations (if available)
    if seamless_hypothesis_translations:
        bleu_seamless = sacrebleu.corpus_bleu(seamless_hypothesis_translations, [gold_translations])
        print(f"{language_code}, seamless, bleu = {bleu_seamless.score:.2f}")
        with open("bleu_score.txt", "a", encoding="utf-8") as f:
            f.write(f"{language_code}, seamless, bleu = {bleu_seamless.score:.2f}\n")
    
    # Calculate BLEU score for nllb translations (if available)
    if nllb_hypothesis_translations:
        bleu_nllb = sacrebleu.corpus_bleu(nllb_hypothesis_translations, [gold_translations])
        print(f"{language_code}, nllb, bleu = {bleu_nllb.score:.2f}")
        with open("bleu_score.txt", "a", encoding="utf-8") as f:
            f.write(f"{language_code}, nllb, bleu = {bleu_nllb.score:.2f}\n")

    # In case both seamless and nllb translations are available, handle both
    if seamless_hypothesis_translations and nllb_hypothesis_translations:
        print(f"{language_code}, both seamless and nllb available.")


