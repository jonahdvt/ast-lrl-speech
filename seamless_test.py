from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import json, os
from config import *   # HF_CODE_MAPPING must map "hi_in" → "hin", etc.

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
    "lg_ug"
    ]
results_dir = "seamless_ft_results"
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
    "lg_ug": "lug"    # Luganda (Uganda)
    }

    model_id = "jonahdvt/seamless-m4t-fleurs-afri"
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
