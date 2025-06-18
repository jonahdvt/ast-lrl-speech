import json
import sacrebleu
from jiwer import wer, Compose, RemoveEmptyStrings, ToLowerCase, RemoveMultipleSpaces, Strip, RemovePunctuation, ReduceToListOfListOfWords
import pandas as pd
import re
import evaluate

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
      mode (str): Which translation mode to evaluate ('seamless', 'seamless_indic', 'nllb', 'gemini', 'openai').
      force (bool): If True, assumes the data is already detokenized and skips detokenization.
    """
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gold_translations = []
    hypothesis_translations = [] # This will hold translations for the specified mode

    # Determine which key to use based on the mode
    mode_to_key_map = {
        "seamless": "seamless_translation",
        "seamless_indic": "translation", # Assuming 'translation' for seamless_indic
        "nllb": "nllb_translation_asr",
        "gemini": "translation", # Assuming 'translation' for gemini
        "openai": "translation"  # Assuming 'translation' for openai
    }
    
    selected_key = mode_to_key_map.get(mode.lower())
    if not selected_key:
        print(f"Error: Invalid mode '{mode}'. Choose from {list(mode_to_key_map.keys())}.")
        return 0.0

    for sample in data:
        gold_text = sample.get('gold_translation')
        hyp_text = sample.get(selected_key)

        if gold_text is None or not gold_text.strip():
            print(f"Warning: Missing or empty gold translation for file {sample.get('file_name', 'Unknown file')}")
            continue

        if hyp_text is None or not hyp_text.strip():
            print(f"Warning: Missing or empty {mode} translation for file {sample.get('file_name', 'Unknown file')}")
            continue
        
        if not force:
            gold_text = detokenize(gold_text)
            hyp_text = detokenize(hyp_text)
        
        gold_translations.append(gold_text)
        hypothesis_translations.append(hyp_text)

    if not hypothesis_translations or not gold_translations:
        print(f"Warning: No valid hypotheses or references to compute BLEU for {language_code} with mode {mode}.")
        return 0.0 # Return 0.0 if no valid pairs

    # sacreBLEU expects references as a list of lists: [[ref1], [ref2], ...]
    bleu_score_result = sacrebleu.corpus_bleu(hypothesis_translations, [gold_translations])
    print(f"{language_code}, {mode}, bleu = {bleu_score_result.score:.2f}")
    
    return bleu_score_result.score


def detailed_wer(json_path):
    """
    Compute detailed WER metrics between Whisper hypothesis and reference (gold) transcripts.
    Supports JSON files where the top level is either:
      - a dict with keys "whisper_transcript" and "gold_transcript", or
      - a list of such dicts (e.g. multiple utterances).
    
    Returns a dict with:
        substitution_rate, deletion_rate, insertion_rate, wer
    all normalized by the total number of reference words.
    """
    # --- helper to compute raw counts for one pair of token lists ---
    def _count_ops(ref_words, hyp_words):
        n_ref = len(ref_words)
        n_hyp = len(hyp_words)
        # DP matrix
        d = [[0]*(n_hyp+1) for _ in range(n_ref+1)]
        for i in range(1, n_ref+1):
            d[i][0] = i
        for j in range(1, n_hyp+1):
            d[0][j] = j
        for i in range(1, n_ref+1):
            for j in range(1, n_hyp+1):
                cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,          # deletion
                    d[i][j-1] + 1,          # insertion
                    d[i-1][j-1] + cost      # substitution or match
                )
        # backtrack
        i, j = n_ref, n_hyp
        subs = dels = ins = 0
        while i>0 or j>0:
            # match
            if i>0 and j>0 and ref_words[i-1]==hyp_words[j-1] and d[i][j]==d[i-1][j-1]:
                i, j = i-1, j-1
            # substitution
            elif i>0 and j>0 and d[i][j]==d[i-1][j-1]+1:
                subs += 1
                i, j = i-1, j-1
            # deletion
            elif i>0 and d[i][j]==d[i-1][j]+1:
                dels += 1
                i -= 1
            # insertion
            elif j>0 and d[i][j]==d[i][j-1]+1:
                ins += 1
                j -= 1
            else:
                # should not occur
                break
        return subs, dels, ins, n_ref

    # --- load JSON ---
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # initialize totals
    total_subs = total_dels = total_ins = total_refs = 0

    # case 1: list of utterances
    if isinstance(data, list):
        for idx, entry in enumerate(data):
            try:
                ref = entry['gold_transcript']
                hyp = entry['whisper_m_ft']
            except (TypeError, KeyError):
                # skip any malformed entry
                continue
            ref_words = ref.split()
            hyp_words = hyp.split()
            s, d, i, n = _count_ops(ref_words, hyp_words)
            total_subs += s
            total_dels += d
            total_ins += i
            total_refs += n

    # case 2: single utterance dict
    elif isinstance(data, dict):
        try:
            ref_words = data['gold_transcript'].split()
            hyp_words = data['whisper_l_ft'].split()
        except (AttributeError, KeyError):
            raise ValueError("JSON must contain 'gold_transcript' and 'whisper_l_ft' keys")
        s, d, i, n = _count_ops(ref_words, hyp_words)
        total_subs, total_dels, total_ins, total_refs = s, d, i, n

    else:
        raise ValueError("Top‐level JSON must be a dict or list of dicts")

    # avoid division by zero
    if total_refs == 0:
        return {
            'substitution_rate': 0.0,
            'deletion_rate': 0.0,
            'insertion_rate': 0.0,
            'wer': float('inf')
        }

    # compute rates
    substitution_rate = round(total_subs / total_refs, 3)
    deletion_rate     = round(total_dels / total_refs, 3)
    insertion_rate    = round(total_ins / total_refs, 3)
    wer               = round((total_subs + total_dels + total_ins) / total_refs, 3)


    return {
        'sub': substitution_rate,
        'del': deletion_rate,
        'ins': insertion_rate,
        'wer': wer
    }

def simple_wer(file, code):
    wer_metric = evaluate.load("wer")
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = []
    references = []

    for sample in data:
        # Using .get() with a default of None is safer for dictionary access
        whisper_pred = sample.get('asr_corrector_pred')
        gold_ref = sample.get('gold_transcript')

        if whisper_pred is None:
            print(f"Warning: Missing whisper transcription for file {sample.get('file_name', 'Unknown file')}")
            continue
        if gold_ref is None:
            print(f"Warning: Missing gold transcription for file {sample.get('file_name', 'Unknown file')}")
            continue

        # Only append when both are present and not empty strings
        if whisper_pred.strip() and gold_ref.strip():
            predictions.append(whisper_pred)
            references.append(gold_ref)
        else:
            print(f"Warning: Empty transcription or reference found for file {sample.get('file_name', 'Unknown file')}")

    if not predictions or not references:
        print(f"Warning: No valid predictions or references to compute WER for {code}.")
        return 0.0 # Return 0.0 if no valid pairs

    wer_score = wer_metric.compute(predictions=predictions, references=references)
    print(f"Final WER for {code}: {wer_score:.4f}")
    return wer_score