import os 
import torch
from datasets import load_dataset
import evaluate
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer
import sacrebleu
import json


def indic_seamless(src_lang, output_json):
    

    dataset = load_dataset("google/fleurs", src_lang, split="test")

    model_name = "ai4bharat/indic-seamless"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    results = []
    
    for idx, sample in enumerate(dataset):
        # Retrieve the file path from the audio sample.
        # The 'audio' field typically contains a dict with a 'path' key.
        file_path = sample["audio"].get("path", f"sample_{idx}.wav")
        file_name = os.path.basename(file_path)
        
        # Extract audio data and sample rate.
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        
        # Process the audio input.
        audio_inputs = processor(audios=audio_array, sampling_rate=sample_rate, return_tensors="pt")
        audio_inputs = {k: v.to("cuda") for k, v in audio_inputs.items()}
        
        # Generate translation.
        output_tokens = model.generate(
            **audio_inputs,
            tgt_lang="eng",
        )
        translated_text = processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        # Append result for current sample.
        results.append({
            "file_name": file_name,
            "translation": translated_text
        })
        
        print(f"Processed {file_name} -> {translated_text}")
    
    # Save the results as a JSON file.
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print("Saved translations to translations.json")
    return results

# Example usage: translate from Hindi to English.



def compute_bleu(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score







def main():
    # Define the languages to process.
    languages = ["hi_in", "te_in", "ta_in", "ml_in"]
    
    for lang in languages:
        pass 









        # Compute BLEU score.
        bleu_score = compute_bleu(predictions, references)
        print(f"BLEU score for {lang}: {bleu_score:.2f}\n")
        

# if __name__ == "__main__":
#    main()

