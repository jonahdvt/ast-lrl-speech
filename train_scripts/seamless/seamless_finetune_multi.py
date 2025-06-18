import os
import torch
import pandas as pd
import evaluate

from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import IterableDataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from mapping import HF_CODE_MAPPING  # your mapping of fleur codes to HF codes

# --- 1. Infinite cycling wrapper for a streaming IterableDataset ---
class InfiniteDataset(IterableDataset):
    def __init__(self, base_iterable):
        self.base = base_iterable
    def __iter__(self):
        while True:
            for ex in self.base:
                yield ex

# --- 2. Streaming loader with lang_code injection ---
def load_streaming_dataset(dataset_name: str, config: str, split: str):
    if "+" in split:
        streams = [
            load_dataset(dataset_name, config, split=s, streaming=True)
            for s in split.split("+")
        ]
        ds = interleave_datasets(streams)
    else:
        ds = load_dataset(dataset_name, config, split=split, streaming=True)
    # inject the config name into every example
    return ds.map(lambda ex: {**ex, "lang_code": config})

# --- 3. Build raw_train and raw_val streams ---
languages = [
    "hi_in", "pa_in", "ta_in", "te_in", "ml_in",
    # "sw_ke", "yo_ng", "ha_ng", "ig_ng", "lg_ug"
]

raw_train = interleave_datasets([
    load_streaming_dataset("google/fleurs", lang, "train")
    for lang in languages
])
raw_val = interleave_datasets([
    load_streaming_dataset("google/fleurs", lang, "validation")
    for lang in languages
])

# --- 4. Load translations CSV and create mapping ---
df = (
    pd.read_csv("fleurs_lang_info/en_translations.csv")
      .rename(columns={"codes": "id", "transcript": "translation"})
      .dropna(subset=["translation"])
)
id2trans = dict(zip(df["id"], df["translation"]))

def add_translation(ex: Dict[str, Any]) -> Dict[str, Any]:
    ex["translation"] = id2trans.get(ex["id"], "")
    return ex

# --- 5. Load processor and define prepare_dataset ---
model_id = "facebook/seamless-m4t-v2-large"
processor = AutoProcessor.from_pretrained(
    model_id, task="translate", trust_remote_code=True
)

SMLS_MAP = {
    "hi_in": "hin", "pa_in": "pan", "ta_in": "tam", "te_in": "tel", "ml_in": "mal", 
    
    "sw_ke": "swh", "ha_ng": "hau", "yo_ng": "yor", "ig_ng": "ibo", "lg_ug": "lug",
}

def prepare_dataset(ex: Dict[str, Any]) -> Dict[str, Any]:
    smls = SMLS_MAP[ex["lang_code"]]
    processor.tokenizer.src_lang = smls
    processor.tokenizer.tgt_lang = "eng"
    audio = ex["audio"]
    inp = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    lbl = processor.tokenizer(ex["translation"]).input_ids
    return {"input_features": inp, "labels": lbl}

# --- 6. Apply the maps before wrapping in InfiniteDataset ---
raw_train = raw_train.map(add_translation)
raw_train = raw_train.map(prepare_dataset)
raw_val   = raw_val.map(add_translation)
raw_val   = raw_val.map(prepare_dataset)

# --- 7. Wrap train set in InfiniteDataset; leave val finite ---
fleurs_train = InfiniteDataset(raw_train)
fleurs_val   = raw_val

# --- 8. Data collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: AutoProcessor
    decoder_start_token_id: int
    max_length: int = 384

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        inputs = [{"input_features": f["input_features"]} for f in features]
        lbls   = [{"input_ids": f["labels"][: self.max_length]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        lbatch = self.processor.tokenizer.pad(lbls, return_tensors="pt", padding=True)
        labels = lbatch["input_ids"].masked_fill(lbatch["attention_mask"] == 0, -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# --- 9. Metric setup ---
bleu = evaluate.load("bleu")
def compute_metrics(pred):
    pred_ids, label_ids = pred.predictions, pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"bleu": bleu.compute(predictions=pred_str, references=[[r] for r in label_str])["bleu"]}

# --- 10. TrainingArguments with ignore_data_skip & no auto-push ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./seamless-m4t-fleurs-indic",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    eval_strategy="steps",
    eval_steps=8800,
    warmup_steps=25,
    max_steps=925,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=2200,
    save_total_limit=6,
    metric_for_best_model="bleu",
    greater_is_better=True,
    logging_steps=50,
    ignore_data_skip=True,   # do NOT fast-forward the stream on resume
    push_to_hub=False,       # disable any automatic Hub push during training
    fp16=False,
    adafactor=True,
    learning_rate=2e-5,
    bf16=True,
    weight_decay=0.0,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=fleurs_train,
    eval_dataset=fleurs_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 11. Train or resume ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
last_ckpt = get_last_checkpoint(training_args.output_dir)
if last_ckpt:
    print(f"Resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    print("Starting training from scratch")
    trainer.train()

# --- 12. Manual push to Hub when fully trained ---
trainer.push_to_hub(
    dataset_tags="google/fleurs",
    dataset="FLEURS",
    dataset_args=f"config: {','.join([HF_CODE_MAPPING[l] for l in languages])}, split: test",
    language="multilingual",
    tags=",".join([HF_CODE_MAPPING[l] for l in languages]),
    model_name="Seamless M4T – FLEURS Adrican Multilingual Fine‑tuning",
    finetuned_from=model_id,
    tasks="speech-translation",
)
# 13. Push with your exact kwargs
kwargs = {
    "dataset_tags": "google/fleurs",
    "dataset": "FLEURS",
    "dataset_args": f"config: {','.join([HF_CODE_MAPPING[l] for l in languages])}, split: test",
    "language": "multilingual",
    "tags": ",".join([HF_CODE_MAPPING[l] for l in languages]),
    "model_name": "Seamless M4T – FLEURS Indic Multilingual Fine‑tuning",
    "finetuned_from": model_id,
    "tasks": "speech-translation",
}

trainer.push_to_hub(**kwargs)
