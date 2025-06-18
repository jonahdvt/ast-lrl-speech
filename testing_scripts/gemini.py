from google import genai
from datasets import load_dataset, Audio
import tempfile
import soundfile as sf
import os
import json

language_codes = {
    # "Hindi":     "hi_in",
    # "Punjabi":   "pa_in",
    # "Tamil":     "ta_in",
    # "Telugu":    "te_in",
    # "Malayalam": "ml_in",

    # "Swahili":   "sw_ke",
    # "Yoruba":    "yo_ng",
    # "Hausa":     "ha_ng",
    # "Igbo":      "ig_ng",
    # "Luganda":   "lg_ug",

    "French":    "fr_fr",
}


client = genai.Client(api_key=GOOGLE_API_KEY)

results_dir = "RESULTS/gemini_results"
os.makedirs(results_dir, exist_ok=True)

for language_name, code in language_codes.items():
    print(f"\n→ Processing {language_name} ({code})…")


    ds = load_dataset(
    "google/fleurs",
    code,
    split="test",
    trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    out_path = os.path.join(results_dir, f"{code}.json")
    results = []

    tmp_dir = tempfile.mkdtemp(prefix="fleurs_audio_")

    for sample in ds:
        arr, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        file_name = os.path.basename(sample["audio"]["path"])

        # write out to a real .wav in tmp_dir
        tmp_path = os.path.join(tmp_dir, file_name)
        sf.write(tmp_path, arr, sr)

        upload_ref = client.files.upload(file=tmp_path)         

        resp = client.models.generate_content(
            model="gemini-2.0-flash",

            contents=[
                f"Given this audio in {language_name}, provide the exact English translation:",
                upload_ref
            ]
        )
        results.append({
            "file_name":    file_name,
            "translation":  resp.text.strip()
        })


    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    print(f"  ↳ saved {len(results)} translations to {out_path}")
