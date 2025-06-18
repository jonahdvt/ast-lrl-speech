from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers.trainer_utils import get_last_checkpoint
import threading
import shutil
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)
from mapping import (
    FLEURS_LANGUAGE_CODES,
    WHISPER_LANGUAGE_CODE_MAPPING,
    HF_CODE_MAPPING,
)





# 1. Setup
# Specify the target language code for FLEURS
languages = [
    # "hi_in", 
    # "pa_in", 
    # "ta_in", 
    # "te_in", 
    # "ml_in",
# #     ]
# # # languages= [
    # "ig_ng",
    # "lg_ug",
    # "sw_ke", 
    # "yo_ng", 
    "fr_fr"
    # "ha_ng",
    ]

# per‐language “base counts”
_base_iters = {
    "hi_in":   2120,
    "pa_in":   1923,
    "ta_in":   2367,
    "te_in":   2302,
    "ml_in":   3043,
    "sw_ke":   3070,
    "ha_ng":   3259,
    "yo_ng":   2339,
    "ig_ng":   2839,
    "lg_ug":   2478,
    "fr_fr":   2584,
}

# scale factor
_SCALE = 0.3125
print("list of langs = ", languages)
for language_code in languages:
    print("training langauge ", language_code)
    whisper_model = 'openai/whisper-large-v3'
    # whisper_model = "jonahdvt/whisper-fleurs-large-afri"
    # whisper_model = "jonahdvt/whisper-fleurs-large-afri"
    MAX_ITER = int(_base_iters[language_code] * _SCALE)

    # 2. Load & stream the FLEURS dataset for the specified language
    fleurs_train = load_dataset(
        "google/fleurs",
        language_code,
        split="train",
        streaming=True,
        trust_remote_code=True
    )


    # Ensure audio is 16kHz
    fleurs_train = fleurs_train.cast_column("audio", Audio(sampling_rate=16000))

    
    lang_token = WHISPER_LANGUAGE_CODE_MAPPING[language_code]

    # …but if we're doing Luganda (lg_ug), force Shona (sn)
    if language_code == "lg_ug":
        lang_token = "Shona"

    # …and if we're doing Igbo (ig_ng), force Lingala (ln)
    elif language_code == "ig_ng":
        lang_token = "Lingala"


    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        whisper_model,
        language=lang_token,
        task="transcribe"
    )
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 4. Preprocessing function
    def prepare_dataset(batch):
        audio = batch["audio"]  # contains 'array' and 'sampling_rate'
        # extract log-mel features
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        # tokenize transcription
        batch["labels"] = processor.tokenizer(
        batch["transcription"],
        truncation=True,
        max_length=processor.tokenizer.model_max_length).input_ids


        return batch

    # 5. Map preprocessing over the streamed datasets
    fleurs_train = fleurs_train.map(
        prepare_dataset,
        remove_columns=["audio", "transcription"]
    )


    # 6. Load pre-trained Whisper for fine-tuning
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)

    # 7. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./whisper-fleurs-large-{language_code}",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        eval_strategy="no",
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=MAX_ITER,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="steps",
        save_steps=MAX_ITER,
        save_total_limit=3,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=50,
        push_to_hub=True,
        hub_strategy="end",
        overwrite_output_dir=False,
    )

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
    collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor, 
            decoder_start_token_id=model.config.decoder_start_token_id,
            max_length=448)

    # 8. Trainer & metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        preds = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        labels = pred.label_ids
        labels[labels == -100] = processor.tokenizer.pad_token_id
        refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=preds, references=refs)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=fleurs_train,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # 9. Train & push
    last_ckpt = get_last_checkpoint(training_args.output_dir)
    if last_ckpt:
        print("step 2 for ", language_code)
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("step 2 for ", language_code)
        trainer.train()



        # 11. Push to Hub
        kwargs = {
            "dataset_tags": "google/fleurs",
            "dataset": "FLEURS",
            "dataset_args": f"config: {HF_CODE_MAPPING[language_code]}, split: train",
            "language": HF_CODE_MAPPING[language_code],
            "model_name": f"Whisper Large FLEURS – {HF_CODE_MAPPING[language_code]}",
            "finetuned_from": whisper_model,
            "tasks": "automatic-speech-recognition",
        }

        model.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        trainer.save_model(training_args.output_dir)
        trainer.push_to_hub(**kwargs)




