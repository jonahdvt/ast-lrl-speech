import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Simple script to push a local Whisper model & processor to the Hugging Face Hub
parser = argparse.ArgumentParser(description="Push Whisper model & processor to HF Hub")
parser.add_argument("--local-dir", required=True, help="Directory with local model artifacts")
parser.add_argument("--repo-id", required=True, help="Target Hub repo (e.g., username/model-name)")
parser.add_argument("--token", default=None, help="HF auth token (or set HF_TOKEN env var)")
args = parser.parse_args()

# Load the trained model and processor from the local directory
model = WhisperForConditionalGeneration.from_pretrained(args.local_dir)
processor = WhisperProcessor.from_pretrained(args.local_dir)

# Push model and processor to the Hub
model.push_to_hub(args.repo_id, use_auth_token=args.token)
print(f"✅ Model weights pushed to '{args.repo_id}'")
processor.push_to_hub(args.repo_id, use_auth_token=args.token)
print(f"✅ Processor/tokenizer pushed to '{args.repo_id}'")

