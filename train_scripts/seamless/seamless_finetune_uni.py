from datasets import load_dataset, Audio, Dataset
from mapping import HF_CODE_MAPPING  # Ensure your config includes necessary mappings for target languages
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import pandas as pd 


# 1. Setup
# Specify the target language codes for FLEURS (e.g., Indic languages)
languages = [
    "hi_in", 
    # "pa_in", 
    # "ta_in", 
    # "te_in", 
    # "ml_in"
    ]
# Alternatively, for African languages:
# languages = [
#     "ig_ng",
#     "lg_ug",
#     "sw_ke", 
#     "yo_ng", 
#     "ha_ng"
# ]



csv_path = "fleurs_lang_info/en_translations.csv"
df = pd.read_csv(csv_path)
df = df.rename(columns={"codes": "id", "transcript": "translation"})
df = df.dropna(subset=["translation"])
translations = Dataset.from_pandas(df[["id", "translation"]])
id2trans = dict(zip(df["id"], df["translation"]))


for language_code in languages:


    smls_map = {
    "hi_in": "hin",  
    "pa_in": "pan",  
    "ta_in": "tam",  
    "te_in": "tel",   
    "ml_in": "mal",  
    "sw_ke": "swh",   
    "ha_ng": "hau",  
    "yo_ng": "yor",  
    "ig_ng": "ibo",  
    "lg_ug": "lug"    
    }

    model_id = "facebook/seamless-m4t-v2-large"

    fleurs_train = load_dataset("google/fleurs", language_code, split="train")
    fleurs_val   = load_dataset("google/fleurs", language_code, split="validation")
    for ds in (fleurs_train, fleurs_val):
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))


    def add_translation(example):
        example["translation"] = id2trans.get(example["id"], "")
        return example

    fleurs_train = fleurs_train.map(add_translation)
    fleurs_val   = fleurs_val.map(add_translation)


    src_code = smls_map[language_code],
    tgt_code = "eng"

    processor = AutoProcessor.from_pretrained(
        model_id,
        task="translate",
        src_lang=src_code,
        tgt_lang=tgt_code,
        trust_remote_code=True,
    )

    # 3. Preprocessing the Dataset
    def prepare_dataset(batch):
        audio = batch["audio"]  # contains 'array' and 'sampling_rate'
        # Compute input features from the audio
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        # For speech translation, assume the dataset has a "translation" field containing the target text.
        batch["labels"] = processor.tokenizer(batch["translation"]).input_ids
        return batch

    # Apply preprocessing to the training and evaluation datasets
    keep = ["audio", "translation"]
    fleurs_train = fleurs_train.map(prepare_dataset, remove_columns=[c for c in fleurs_train.column_names if c not in keep])
    fleurs_val   = fleurs_val.map(prepare_dataset,   remove_columns=[c for c in fleurs_val.column_names   if c not in keep])



    fleurs_train.cleanup_cache_files()
    fleurs_val.cleanup_cache_files()

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./seamless-m4t-v2-large-{language_code}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        eval_strategy="steps",
        learning_rate=1e-5,
        warmup_steps=100,
        num_train_epochs=10,
        gradient_checkpointing=True,
        eval_accumulation_steps=4,
        fp16=True,
        logging_steps=100,
        eval_steps=100000,
        save_steps=100000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        push_to_hub=True,
        overwrite_output_dir=True,
    )

    # 5. Defining Data Collator and BLEU Metric for Evaluation
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int
        max_length: int = 448  # Maximum label length

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Separate the input features and labels for different padding strategies
            input_features = [{"input_features": f["input_features"]} for f in features]
            label_features = [{"input_ids": f["labels"][:self.max_length]} for f in features]

            # Pad the audio inputs
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            # Pad the labels
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)
            # Replace padding token id with -100 so that it is ignored during loss computation
            labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"] == 0, -100)
            # Remove the BOS token if present at the beginning of all sequences
            if (labels[:, 0] == self.decoder_start_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, 
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_length=448
    )

    bleu_metric = evaluate.load("bleu")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # Replace -100 in labels with the pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        # Decode the predictions and labels to text
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # For BLEU, each reference should be provided as a list
        bleu_score = bleu_metric.compute(predictions=pred_str, references=[[ref] for ref in label_str])
        return {"bleu": bleu_score["bleu"]}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=fleurs_train,
        eval_dataset=fleurs_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Train the model. The best model (highest BLEU on eval) will be loaded at the end.
    trainer.train()

    kwargs = {
        "dataset_tags": "google/fleurs",
        "dataset": "FLEURS",  
        "dataset_args": f"config: {HF_CODE_MAPPING[language_code]} -> en, split: test",
        "language": HF_CODE_MAPPING[language_code],
        "model_name": f"Seamless M4T – {HF_CODE_MAPPING[language_code]} FLEURS Fine‑tuning",
        "finetuned_from": model_id,
        "tasks": "speech-translation",
    }

    # Save and push the model, its configuration, and training arguments to the Hugging Face Hub
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    trainer.push_to_hub(**kwargs)

    # To load the model later, you can use:
    # from transformers import pipeline
    # st_pipeline = pipeline(
    #     "speech-translation",
    #     model="your-username/your-model-name",
    #     tokenizer="your-username/your-model-name",
    #     feature_extractor="your-username/your-model-name",
    # )
