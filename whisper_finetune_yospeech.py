#!/usr/bin/env python3
"""
finetune_whisper_yo.py

Fine-tunes Whisper Large v3 on Yoruba speech for 1 hour (~831 samples), interleaved
with FLEURS Yoruba data, using Hugging Face Transformers and Datasets streaming.

Usage:
  pip install datasets transformers evaluate soundfile
  python finetune_whisper_yo.py
"""
import itertools
import torch
from dataclasses import dataclass
from typing import List, Dict, Any

from datasets import load_dataset, Audio, interleave_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

# 1. Dataset settings for Yoruba
max_yo_samples = 831               # ~1 hour of Yoruba speech
fleurs_lang = "yo_ng"             # FLEURS Yoruba
max_fleurs_samples = None          # use full FLEURS Yoruba split

# 2. Load and normalize streams
yo_ds = (
    load_dataset("yo-speech/yo-asr-speech", split="train", streaming=True)
    .take(max_yo_samples)
    .cast_column("audio", Audio(sampling_rate=16_000))
    .map(
        lambda batch: {
            "audio_array": batch["audio"]["array"],
            "sampling_rate": batch["audio"]["sampling_rate"],
            "sentence": batch.get("sentence", batch.get("text", "")),
        },
        remove_columns=[c for c in load_dataset("yo-speech/yo-asr-speech", split="train").column_names
                        if c not in ("audio", "sentence")],
        batched=False,
    )
)

fle_ds = (
    load_dataset("google/fleurs", name=fleurs_lang, split="train", streaming=True)
    .cast_column("audio", Audio(sampling_rate=16_000))
    .map(
        lambda batch: {
            "audio_array": batch["audio"]["array"],
            "sampling_rate": batch["audio"]["sampling_rate"],
            "sentence": batch["transcription"],
        },
        remove_columns=[c for c in load_dataset("google/fleurs", name=fleurs_lang, split="train").column_names
                        if c not in ("audio", "transcription")],
        batched=False,
    )
)

# 3. Interleave Yoruba and FLEURS streams
train_ds = interleave_datasets([yo_ds, fle_ds])

# 4. Whisper processor & tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", task="transcribe")
processor = WhisperProcessor(feature_extractor, tokenizer)
# Force language token to Yoruba
processor.tokenizer.language = "yoruba"

# 5. Prepare inputs
train_ds = train_ds.map(
    lambda batch: {
        "input_features": processor.feature_extractor(
            batch["audio_array"], sampling_rate=batch["sampling_rate"]
        ).input_features[0],
        "labels": processor.tokenizer(batch["sentence"]).input_ids,
    },
    remove_columns=["audio_array", "sampling_rate", "sentence"],
    batched=False,
)

# 6. Load model & define training args
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-yo-ys-1h",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    eval_strategy="no",
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=990,
    gradient_checkpointing=True,
    fp16=True,
    save_strategy="steps",
    save_steps=990,
    save_total_limit=3,
    metric_for_best_model="wer",
    greater_is_better=False,
    logging_steps=50,
    push_to_hub=True,
    hub_strategy="end",
    overwrite_output_dir=False,
)

# 7. Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int
    padding: bool = True
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_feats = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]
        batch_inputs = self.processor.feature_extractor.pad(
            {"input_features": input_feats}, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels}, return_tensors="pt", padding=True
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"] == 0, self.label_pad_token_id
        )
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": batch_inputs["input_features"], "labels": labels}

collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

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
    train_dataset=train_ds,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# 9. Train & push
last_ckpt = get_last_checkpoint(training_args.output_dir)
if last_ckpt:
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    trainer.train()

trainer.push_to_hub(
    dataset_tags="yo-speech",
    dataset="yo-asr-speech",
    dataset_args=f"1h (~{max_yo_samples} samples)",
    language="yo",
    model_name="Whisper Large — Yoruba - Yoruba Speech - 1h",
    finetuned_from="openai/whisper-large-v3",
    tasks="automatic-speech-recognition",
)

# 10. Save locally & push processor
model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
processor.push_to_hub("jonahdvt/whisper-large-yo-ys-1h")
