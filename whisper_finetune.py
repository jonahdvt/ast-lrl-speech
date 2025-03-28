from datasets import load_dataset, Audio
from config import FLEURS_LANGUAGE_CODES, WHISPER_LANGUAGE_CODE_MAPPING, HF_CODE_MAPPING
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union




# 1. Setup
# Specify the target language code for FLEURS
languages = ["yo_ng", "ha_ng", "sw_ke"]

language_code = "yo_ng"  
# whisper_model = 'openai/whisper-small'
whisper_model = "jonahdvt/whisper-fleurs-ha_ng"


# Load the FLEURS dataset for the specified language
# Combine train + validation splits for training
fleurs_train = load_dataset("google/fleurs", language_code, split="train")
fleurs_val = load_dataset("google/fleurs", language_code, split="validation")
fleurs_test  = load_dataset("google/fleurs", language_code, split="test")

# Ensure audio is 16kHz
fleurs_train = fleurs_train.cast_column("audio", Audio(sampling_rate=16000))
fleurs_val = fleurs_val.cast_column("audio", Audio(sampling_rate=16000))
fleurs_test  = fleurs_test.cast_column("audio", Audio(sampling_rate=16000))


# 2. Preparing the Feature Extractor and Tokenizer
# Load feature extractor and tokenizer for Whisper
feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
tokenizer = WhisperTokenizer.from_pretrained(whisper_model, language=WHISPER_LANGUAGE_CODE_MAPPING[language_code], task="transcribe")
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# 3. Preprocessing the Dataset
def prepare_dataset(batch):
    audio = batch["audio"]  # contains 'array' and 'sampling_rate'
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]     # Compute log-Mel spectrogram features from audio
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids    # Encode transcription text to label ids
    return batch

# Apply the preprocessing to the training and test data. Each has input features(audio) and labels (transcript)
fleurs_train = fleurs_train.map(prepare_dataset, remove_columns=fleurs_train.column_names)
fleurs_val = fleurs_val.map(prepare_dataset, remove_columns=fleurs_val.column_names)
fleurs_test = fleurs_test.map(prepare_dataset, remove_columns=fleurs_test.column_names)
# Removed setting num_proc=4 


fleurs_train.cleanup_cache_files()
fleurs_val.cleanup_cache_files()
fleurs_test.cleanup_cache_files()


# 4. Training config 
# load the pre-trained Whisper model for fine-tuning
model = WhisperForConditionalGeneration.from_pretrained(whisper_model)


"""
    - Batch size: Whisper models are large; a common batch size is 16 (as in the Hindi fine-tuning example)​
        Adjust if you encounter memory issues (you can decrease batch size and increase gradient_accumulation_steps to compensate​).

    - Learning rate: A small learning rate (e.g. 1e-5) is recommended for fine-tuning Whisper​

    - FP16: Enable mixed-precision training (fp16=True) for faster training if using a GPU with half-precision support.

    - Gradient checkpointing: gradient_checkpointing=True can reduce memory usage at the cost of some speed​.

    - Evaluation strategy: We evaluate periodically (e.g., every certain number of steps) to monitor WER on the validation/test set.

    - Save and logging: Set save_steps, eval_steps, and logging_steps as needed. 
        Also, load_best_model_at_end=True with metric_for_best_model="wer" ensures we keep the best checkpoint (lowest WER)
"""
training_args = Seq2SeqTrainingArguments(
    #output_dir="./whisper-fleurs-small-{}".format(language_code),  # save directory (can be changed)
    output_dir="./whisper-fleurs-small-afri", # combined dir
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,  # increase this if reducing batch_size
    eval_strategy="steps",    # evaluate every few steps (set eval_steps)
    learning_rate=1e-5,
    warmup_steps=100,
    num_train_epochs=10,
    gradient_checkpointing=True,
    eval_accumulation_steps=4,
    fp16=True,
    logging_steps=50,
    eval_steps=1000,                
    save_steps=1000,                
    save_total_limit=2,            # only keep last 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,       # lower WER is better
    push_to_hub=True,
    overwrite_output_dir=True,
)



# 5. Defining Data Collator and WER Metric for Evaluation

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int
    max_length: int = 448  # Add a field for maximum label length

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate the inputs and labels since we need to pad them differently
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"][:self.max_length]} for f in features]

        # Pad the audio inputs (returns dict with 'input_features')
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # Pad the labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)
        # Replace padding token id with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"] == 0, -100)
        # If the tokenizer added a BOS token at position 0 for all sequences, remove it (Whisper tokenizer might add BOS)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

# Initialize our data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, 
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_length=448)

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions  # model predictions
    label_ids = pred.label_ids   # true labels
    # Replace -100 in labels as we set before
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    # Decode to text
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=fleurs_train,
    eval_dataset=fleurs_val,            # using test set for evaluation (could also use validation)
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,  # this ensures the Trainer pads the features correctly
)
# After training, the best model (lowest WER on eval) will be loaded 
trainer.train()

# kwargs = {
#     "dataset_tags": "google/fleurs",
#     "dataset": "FLEURS",  
#     "dataset_args": f"config: {HF_CODE_MAPPING[language_code]}, split: test",
#     "language": HF_CODE_MAPPING[language_code],
#     "model_name": "Whisper Small FLEURS - Jonah Dauvet",
#     "finetuned_from": "openai/whisper-small",
#     "tasks": "automatic-speech-recognition",
# }
kwargs = {
    "dataset_tags": "google/fleurs",
    "dataset": "FLEURS",
    "dataset_args": f"config: {','.join([HF_CODE_MAPPING[l] for l in languages])}, split: test",
    "language": "multilingual",                     # <— indicates multiple languages
    "tags": ",".join([HF_CODE_MAPPING[l] for l in languages]),  # <— use tags to list individual codes
    "model_name": "Whisper Small FLEURS – African Fine‑tuning",
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

# This call will push model, its configuration, and training arguments to the HF Hub. load it using the model identifier.
model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
trainer.save_model(training_args.output_dir)
trainer.push_to_hub(**kwargs)


# Can use to load with       processor = WhisperProcessor.from_pretrained("your-username/your-model-name")
# Retrieve by 
# from transformers import pipeline
# asr_pipeline = pipeline(
#     "automatic-speech-recognition",
#     model="your-username/your-model-name",
#     tokenizer="your-username/your-model-name",
#     feature_extractor="your-username/your-model-name",



