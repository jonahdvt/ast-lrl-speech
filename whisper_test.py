from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline,WhisperFeatureExtractor,WhisperTokenizer
import evaluate
from config import *

# 1. Setup
language_code = "pa_in"
# Path to the directory where the model & processor were saved (or the Hub identifier)
model_dir = f"./whisper-fleurs-small-{language_code}"

# 2. Load the FLEURS test dataset
fleurs_test = load_dataset("google/fleurs", language_code, split="test")
fleurs_test = fleurs_test.cast_column("audio", Audio(sampling_rate=16000))

# 3. Load the trained model and processor
#processor = WhisperProcessor.from_pretrained(model_dir)
#model = WhisperForConditionalGeneration.from_pretrained(model_dir)

# 3b. Load the classic model and processor
model = 'openai/whisper-small'
feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
tokenizer = WhisperTokenizer.from_pretrained(model, language=WHISPER_LANGUAGE_CODE_MAPPING[language_code], task="transcribe")
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# 4. (Optional) Create a Hugging Face pipeline for ASR
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True
)

# 5. Evaluate the model on the test set
wer_metric = evaluate.load("wer")
predictions = []
references = []
count = 0

# Iterate over test samples (this example uses a simple loop; for large datasets consider batching)
for sample in fleurs_test:
    count+=1
    # Use the pipeline to obtain transcription
    result = asr_pipeline(sample["audio"]["array"])
    if "Whisper did not predict an ending timestamp" in result["text"]:
        print("Skipping sample due to incomplete timestamp.")
        continue  # Skip this sample

    predictions.append(result["text"])
    print(count/len(fleurs_test))
    references.append(sample["transcription"])

wer = wer_metric.compute(predictions=predictions, references=references)
print("Final WER on FLEURS test set:", wer)



# Whisper did not predict an ending timestamp, which can happen if audio is cut off in the middle of a word. Also make sure WhisperTimeStampLogitsProcessor was used during generation.
# Error message in the weird ones at Actual.


# SMALL
# Final WER on Yoruba FLEURS test set:              0.6766211180124223
# UNTRAINED Final WER on Yoruba FLEURS test set:    1.778583850931677

# Final WER on Malayalam FLEURS test set:           0.511
# UNTRAINED Final WER on Malayalam FLEURS test set: 4.271


# Final WER on Telugu FLEURS test set:           0.531
# UNTRAINED Final WER on Telugu FLEURS test set: 1.0527930461459798


# Final WER on Punjabi FLEURS test set:           0.460
# UNTRAINED Final WER on Punjabi FLEURS test set: 


#MED 
# Final WER on Yoruba FLEURS test set: 0.6451180124223602
