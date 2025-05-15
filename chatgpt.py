import io
import os
import base64
import soundfile as sf
import json
import re

from datasets import load_dataset
from openai import OpenAI

client = OpenAI()
os.makedirs("openai_results", exist_ok=True)
language_codes = [
    # "sw_ke",
    # "ha_ng",
    # "yo_ng",
    # "lg_ug",
    # "ig_ng",
    # "hi_in",
    # "pa_in", 
    # "ta_in", 
    # "te_in", 
    "ml_in",
    
    ]
for l in language_codes:

    ds = load_dataset("google/fleurs", l, split="test", streaming=True)

    transcripts = []

    for i, sample in enumerate(ds, start=1):
        arr = sample["audio"]["array"]
        sr  = sample["audio"]["sampling_rate"]
        path = sample["audio"]["path"]  

        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        wav_bytes = buf.getvalue()
        buf.close()

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Translate this audio from {l} to English."},
                    {"type": "input_audio", "input_audio": {
                        "data": audio_b64,
                        "format": "wav"
                    }}
                ]
            }]
        )
        # print(completion.choices[0])
        # choice_str = str(completion.choices[0])
        # m = re.search(r"transcript='([^']*)'", choice_str)
        # if m:
        #     transcript = m.group(1)
        audio_response = completion.choices[0].message.audio
        transcript      = audio_response.transcript
        print(transcript)


        transcripts.append({
            "file_name": path,
            "translation": transcript
        })

    out_path = os.path.join("openai_results", f"{l}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)

