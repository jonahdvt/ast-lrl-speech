from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import json, os
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)
from mapping import *

language_codes = [
    # "hi_in",
    # "pa_in", 
    # "ta_in", 
    # "te_in", 
    # "ml_in",
    # "sw_ke",
    # "ha_ng",
    # "yo_ng",
    # "ig_ng",
    # "lg_ug"
    "fr_fr"
    ]
results_dir = "/home/mila/d/dauvetj/mila-speech-2/RESULTS/s_seamless_baseline"
os.makedirs(results_dir, exist_ok=True)

for language_code in language_codes:
    smls_map = {
    "hi_in": "hin",   # Hindi (India)
    "pa_in": "pan",   # Punjabi (India)
    "ta_in": "tam",   # Tamil (India)
    "te_in": "tel",   # Telugu (India)
    "ml_in": "mal",   # Malayalam (India)
    "sw_ke": "swh",   # Swahili (Kenya)
    "ha_ng": "hau",   # Hausa (Nigeria)
    "yo_ng": "yor",   # Yoruba (Nigeria)
    "ig_ng": "ibo",   # Igbo (Nigeria)
    "lg_ug": "lug",    # Luganda (Uganda)
    "fr_fr": "fra"   # French (France)
    }

    model_id = "facebook/seamless-m4t-v2-large"
    src_code = smls_map[language_code]
    tgt_code = "eng"

    # 1) Load processor & model
    processor = AutoProcessor.from_pretrained(
        "facebook/seamless-m4t-v2-large",
        task="translate",
        src_lang=src_code,
        tgt_lang=tgt_code,
        trust_remote_code=True,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, trust_remote_code=True
    )

    # 2) Load FLEURS test set, resampling audio to 16 kHz
    fleurs_test = load_dataset(
        "google/fleurs", language_code, split="test", trust_remote_code=True
    ).cast_column("audio", Audio(sampling_rate=16_000))

    results = []
    for idx, sample in enumerate(fleurs_test, start=1):
        # a) Turn raw audio into model inputs
        inputs = processor(
            audios=sample["audio"]["array"],
            sampling_rate=16_000,
            return_tensors="pt",
        )

        # b) Generate translated-text tokens
        generated = model.generate(
            **inputs,
            tgt_lang=tgt_code,
            max_new_tokens=256,   # or whatever you prefer
        )

        # c) Decode to a Python string
        # generated shape is [batch_size=1, seq_len]
        token_ids = generated[0].tolist()
        translation = processor.decode(
            generated[0].tolist(),
            skip_special_tokens=True
        )

        # d) Record result
        file_id = sample["audio"].get("path", f"sample_{idx}")
        results.append({
            "file_name": file_id,
            "seamless_translation": translation
        })


        # if idx % 50 == 0:
        #     print(f"→ Processed {idx} samples…")

    # 3) Save to JSON
    outpath = os.path.join(results_dir, f"{language_code}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Done {len(results)} samples for {language_code}. Saved to {outpath}")
