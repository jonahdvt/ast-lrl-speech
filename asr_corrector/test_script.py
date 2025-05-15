import os
import json
import glob
import evaluate
import pandas as pd
from data_augmentation import WERMetric

lang_dict = {
    "yo_ng": "yor",
    "ha_ng": "hau",
    "sw_ke": "swa",
    "lg_ug": "lug",
    "ig_ng": "ibo",
    "hi_in": "hin",
    "ml_in": "mlm",
    "pa_in": "pan",
    "ta_in": "tam",
    "te_in": "tel"}


def calculate_baselines():
    results = {}
    metric = evaluate.load("sacrebleu")
    for region in ["afri", "indic"]:
        files = glob.glob(f'test_data/{region}/*.json')

        for file in files:
            language = file.split('/')[-1].replace('.json', '')
            iso1 = language.split("_")[0]
            iso3 = lang_dict[language]
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            reference = [entry["translation"][iso3].strip() for entry in data if iso3 in entry["translation"]]
            prediction = [entry["translation"][iso1].strip() for entry in data if iso1 in entry["translation"]]
            bleu_ref = [[i] for i in reference]

            wer_metric = WERMetric()
            _, score = wer_metric.calculate(reference, prediction)

            bleu = metric.compute(predictions=prediction, references=bleu_ref)["score"]
            results[language] = {"bleu": round(bleu, 4), "wer": round(score, 4)}

            outpath = "test_data/per_lang_results.json"
            with open(outpath, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"{language} Bleu Score: {bleu}")


def add_predictions_to_json(json_path, preds_path, output_dir=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(preds_path, "r", encoding="utf-8") as f:
        preds = [line.strip() for line in f if line.strip()]

    print(f"Data Length: {len(data)} \n Pred Length: {len(preds)}")

    assert len(data) == len(preds), "Mismatch between number of JSON entries and predictions"

    for entry, pred in zip(data, preds):
        entry["asr_corrector_pred"] = pred

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(json_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collate_predictions():
    data_dir = "../ft_whisper_results/l_ft_whisper_results"
    pred_dir = "predictions"

    json_files = glob.glob(os.path.join(data_dir, "*/*.json"))
    pred_files = glob.glob(os.path.join(pred_dir, "*_preds.txt"))

    lang_map = {v: k for k, v in lang_dict.items()}

    for pred_file in pred_files:
        print(pred_file)
        lang_code = os.path.basename(pred_file).replace("_preds.txt", "")
        matching_json = [
            f for f in json_files if lang_map[lang_code] in
                                     os.path.basename(f).replace("_afri.json", "").replace("_indic.json", "")
        ]
        json_path = matching_json[0]
        print(f"Adding predictions for {lang_code} -> {json_path}")
        add_predictions_to_json(json_path, pred_file, "asr_corrector_prediction")


def collate_preds(output_dir=None):
    data_dir = f"test_data"

    for region in ["afri", "indic"]:
        filepath = os.path.join(data_dir, f"{region}_transcripts_test.csv")

        df = pd.read_csv(filepath)
        for lang in lang_dict.keys():
            lang_df = df[df["language"] == lang]
            if lang_df.empty:
                continue

            pred_path = f"predictions/{lang_dict[lang]}_preds.txt"
            with open(pred_path, "r", encoding="utf-8") as f:
                preds = [line.strip() for line in f if line.strip()]

            assert len(lang_df) == len(preds), "Mismatch between number of dataframe and predictions"
            lang_df['asr_corrector_pred'] = preds

            os.makedirs(output_dir, exist_ok=True)
            outpath = os.path.join(output_dir, f"{lang}.json")

            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(lang_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

collate_preds("prediction_json")
