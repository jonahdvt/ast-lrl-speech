from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor


CV_TRANSCRIPT_URL = "https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/tree/main/transcript"

CV_AUDIO_URL = "https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/tree/main/audio"

CV_LANG_CODES = [ # TELUGU AND YORUBA NOT IN COMMON VOICE
    "ha",
    "hi",
    "ig",
    "lg",
    "ml",
    "pa", 
    "sw",
    "ta"
]


common_voice = DatasetDict()

# for l in CV_LANG_CODES:
#     common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", CV_LANG_CODES[l], split="train+validation")
#     common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", CV_LANG_CODES[l], split="test")

#     print(common_voice)

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)


processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
print(common_voice["train"][0])




def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

