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

# Language and dataset settings
cv_lang = "sw"     
fleurs_lang = "sw_ke"
max_cv_samples = 1750
max_fleurs_samples = None

# 1. Load, normalize, and flatten Common Voice streaming

def load_commonvoice_stream(lang, split="validated", max_samples=1750):
    actual_split = "other" if (lang == "ig" and split == "validated") else split
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        name=lang,
        split=actual_split,
        streaming=True,
    )
    if max_samples:
        ds = ds.take(max_samples)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    def normalize_cv(batch: Dict[str, Any]) -> Dict[str, Any]:
        # Flatten audio dict into arrays
        arr = batch["audio"]["array"]
        sr = batch["audio"]["sampling_rate"]
        return {"audio_array": arr, "sampling_rate": sr, "sentence": batch["sentence"]}
    return ds.map(
        normalize_cv,
        remove_columns=[c for c in ds.column_names if c not in ("audio", "sentence")],
        batched=False,
    )

# 2. Load, normalize, and flatten FLEURS streaming
def load_fleurs_stream(lang="sw_ke", split="train", max_samples=None):
    ds = load_dataset(
        "google/fleurs",
        name=lang,
        split=split,
        streaming=True,
    )
    if max_samples:
        ds = ds.take(max_samples)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    def normalize_fleurs(batch: Dict[str, Any]) -> Dict[str, Any]:
        arr = batch["audio"]["array"]
        sr = batch["audio"]["sampling_rate"]
        return {"audio_array": arr, "sampling_rate": sr, "sentence": batch["transcription"]}
    return ds.map(
        normalize_fleurs,
        remove_columns=[c for c in ds.column_names if c not in ("audio", "transcription")],
        batched=False,
    )

# 3. Prepare streams
cv_ds = load_commonvoice_stream(cv_lang, max_samples=max_cv_samples)
fle_ds = load_fleurs_stream(fleurs_lang, max_samples=max_fleurs_samples)
# 4. Interleave flattened streams
commonvoice_flat = interleave_datasets([cv_ds, fle_ds])

# 5. Setup Whisper processor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", task="transcribe")
processor = WhisperProcessor(feature_extractor, tokenizer)
processor.tokenizer.language = "sn"

# 6. Map interleaved flat into model-ready inputs
def prepare_dataset(batch: Dict[str, Any]) -> Dict[str, Any]:
    input_feats = processor.feature_extractor(
        batch["audio_array"],
        sampling_rate=batch["sampling_rate"],
    ).input_features[0]
    label_ids = processor.tokenizer(batch["sentence"]).input_ids
    return {"input_features": input_feats, "labels": label_ids}

commonvoice_train = commonvoice_flat.map(
    prepare_dataset,
    remove_columns=["audio_array", "sampling_rate", "sentence"],
    batched=False,
)

# 7. Model and training arguments
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-sw-2.5h",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    eval_strategy="no",
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1500,
    gradient_checkpointing=True,
    fp16=True,
    save_strategy="steps",
    save_steps=1500,
    save_total_limit=3,
    metric_for_best_model="wer",
    greater_is_better=False,
    logging_steps=50,
    push_to_hub=True,
    hub_strategy="end",
    overwrite_output_dir=False,
)

# 8. Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int
    padding: bool = True
    max_length: int = 448
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
            labels_batch["attention_mask"] == 0,
            self.label_pad_token_id,
        )
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": batch_inputs["input_features"], "labels": labels}

# Instantiate data collator
collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# 9. Metric and trainer. Metric and trainer
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    preds = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=preds, references=refs)}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=commonvoice_train,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# 10. Train and push
last_ckpt = get_last_checkpoint(training_args.output_dir)
if last_ckpt:
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    trainer.train()
trainer.push_to_hub(
    dataset_tags="common_voice",
    dataset="Common Voice",
    dataset_args=f"config: {cv_lang}, split: validated",
    language=cv_lang,
    model_name="Whisper Large — Swahili (2.5h)",
    finetuned_from="openai/whisper-large-v3",
    tasks="automatic-speech-recognition",
)

model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
processor.push_to_hub("jonahdvt/whisper-large-sw-2.5h")