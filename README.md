# Speech Translation Model Training and Evaluation

This repository provides a comprehensive framework for fine-tuning and evaluating speech translation models, including Whisper and Seamless, on the FLEURS dataset. The framework supports both monolingual and multilingual configurations with easy-to-use training and testing scripts.

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Structure](#dataset-structure)
- [Fine-tuning Models](#fine-tuning-models)
- [Testing and Evaluation](#testing-and-evaluation)
- [Results](#results)

## 🚀 Environment Setup


### Installation

```bash
# Clone the repository
git clone https://github.com/jonahdvt/ast-lrl-speech.git
cd ast-lrl-speech

# Install dependencies
conda env create -f environment.yaml
conda activate speech-processing-env
```

## 📊 Dataset Structure

The project uses the FLEURS dataset with the following structure:

### Base Data Location
All dataset files are stored in the `lang_aggregate_data/` directory.

### Dataset Files
- **Language-specific info**: `[langname]_fleurs_info.csv` - Contains metadata for each language
- **English translations**: `en_translation.csv` - Reference translations in English
- **Audio mapping**: Each audio file is mapped to corresponding translations across all languages using the `codes` attribute

### Data Mapping
The `codes` attribute serves as the primary key for mapping audio files to their translations across different languages in the processing scripts.

## 🎯 Fine-tuning Models

### Whisper Fine-tuning

Navigate to the appropriate Whisper fine-tuning script based on your dataset requirements:

```
train_scripts/whisper/whisper_finetune_xxxxx.py
```

#### Configuration Steps:
1. **Select script**: Choose the appropriate fine-tuning script for your target dataset
2. **Language selection**: Specify the languages you want to fine-tune on
3. **Training iterations**: Set the number of iterations (especially important for streaming)
4. **Base model**: Choose the pre-trained Whisper model to fine-tune from
5. **Training arguments**: Configure your desired `training_args`
6. **Model deployment**: Set up `kwargs` for pushing and saving the model to Hugging Face Hub

#### Example Configuration:
```python
languages= [
    "ig_ng",
    "lg_ug",
    "sw_ke", 
    "yo_ng", 
    "ha_ng"
    ]  # Target languages - Example from African multilingual finetuning
whisper_model = 'openai/whisper-large-v3'
```

### Seamless Fine-tuning

Navigate to the appropriate Seamless fine-tuning script based on your configuration:

```
train_scripts/seamless/seamless_finetune_xxxxx.py
```

#### Configuration Options:
- **Mono configuration**: Single language fine-tuning
- **Multi configuration**: Multi-language fine-tuning

#### Configuration Steps:
1. **Select configuration**: Choose between mono/multi configuration scripts
2. **Language selection**: Specify target languages
3. **Training iterations**: Set iteration count for streaming scenarios
4. **Base model**: Select the base Seamless model
5. **Training arguments**: Configure training parameters
6. **Model deployment**: Set up Hugging Face Hub integration in `kwargs`

## 🧪 Testing and Evaluation

### Whisper Testing

```bash
python testing_scripts/whisper_test.py
```

#### Configuration:
- **Model selection**: Choose between base or custom fine-tuned model in `model_id`
- **Language selection**: Select desired languages for direct inference
- **Dataset**: Runs inference on the FLEURS test dataset

### Seamless Testing

```bash
python testing_scripts/seamless_test.py
```

#### Configuration:
- **Model selection**: Choose between base or custom fine-tuned model in `model_id`
- **Language selection**: Select desired languages for direct inference
- **Dataset**: Runs inference on the FLEURS test dataset

### Gemini API Testing

```bash
python testing_scripts/gemini.py
```

#### Requirements:
- Valid Gemini API key
- Configure target languages at the top of the script
- Provides translation inference using Gemini 2.0 Flash on the FLEURS test dataset

### ChatGPT API Testing

```bash
python testing_scripts/chatgpt.py
```

#### Requirements:
- Valid OpenAI API key
- Configure target languages at the top of the script
- Provides translation inference using GPT 4 on the FLEURS test dataset

## 📈 Results

Results from all testing scripts are automatically saved and can be found in the designated output directories. 
To calculate metrics (WER/BLEU), see `metrics.py`
To extract best/worst samples based on WER differencials, see `dataset/sample_finder.py`

## 🔧 Configuration Tips

1. **Memory Management**: Adjust batch sizes based on your GPU memory. Streaming is also advised to avoid downloading large datasets. When training a model, pushing it to Hugging Face Hub is a good way to avoid carrying large models locally. 
2. **Streaming**: Use appropriate iteration counts for streaming datasets
3. **Model Versioning**: Use descriptive names when pushing to Hugging Face Hub
4. **API Limits**: Be mindful of API rate limits when using Gemini or GPT testing

## 📝 Notes

- Ensure all API keys are properly configured before running external API tests
- Monitor training progress and adjust hyperparameters as needed
- Consider using different base models for optimal performance on specific languages

---

For detailed configuration examples and troubleshooting, please refer to the individual script documentation within each file.