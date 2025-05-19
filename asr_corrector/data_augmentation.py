import os
import re
import json
import glob
import werpy
import string
import argparse
import numpy as np
import pandas as pd
from evaluate import load
from torchmetrics.text import CharErrorRate
from typing import List, Dict, Optional, Tuple

pd.set_option('display.max_columns', 4)


def load_all_jsons(json_dir):
    all_rows = []
    for file in glob.glob(json_dir):
        lang = file.split('/')[-1].replace('_afri.json', '').replace('_indic.json','').replace('.json', '')
        data_name = file.split('/')[-2]
        with open(file) as f:
            rows = json.load(f)
            for row in rows:
                row["language"] = lang
                row['data_name'] = data_name
            all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def remove_punctuation(text):
    extra_punctuations = "‘’“”…"
    text = re.sub(r'\s*\[[^\]]+\]\s*', ' ', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', extra_punctuations))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class WERMetric:
    EMPTY_LIST_ERROR_MESSAGE = "The list of predictions and targets cannot be empty"

    def __init__(self):
        self.wer_metric = load("wer")

    def _clean_text(self, text: str) -> str:
        text = remove_punctuation(text)
        text = werpy.normalize(text)
        return text

    def calculate(self, references: List[str], predictions: List[str], normalize=True) -> Tuple[float, float]:
        """
        :param references: str The prediction
        :param predictions: str The reference

        :return: Tuple[float, float] The WER aggregate and the WER score
        """
        if len(references) == 0 or len(predictions) == 0:
            raise Exception(self.EMPTY_LIST_ERROR_MESSAGE)

        if normalize:
            references = [self._clean_text(text) for text in references]
            predictions = [self._clean_text(text) for text in predictions]

        wer_aggregate = werpy.wer(references, predictions)
        wer_vec = werpy.wers(references, predictions)
        score = np.array(wer_vec).mean()
        return wer_aggregate, score


class CERMetric:
    EMPTY_LIST_ERROR_MESSAGE = "The list of predictions and targets cannot be empty"

    def __init__(self):
        self.cer_metric = CharErrorRate()

    def _clean_text(self, text: str) -> str:
        text = remove_punctuation(text)
        text = werpy.normalize(text)
        return text

    def calculate(self, references: List[str], predictions: List[str], normalize=True) -> Tuple[float, float]:
        """
        :param references: str The prediction
        :param predictions: str The reference

        :return: Tuple[float, float] The WER aggregate and the WER score
        """
        if len(references) == 0 or len(predictions) == 0:
            raise Exception(self.EMPTY_LIST_ERROR_MESSAGE)

        if normalize:
            references = [self._clean_text(text) for text in references]
            predictions = [self._clean_text(text) for text in predictions]

        cers = []
        count = 0
        for ref, pred in zip(references, predictions):
            cers.append(self.cer_metric(pred, ref))
            count = count + 1
        score = np.array(cers).mean()
        return score


def build_data(test=True):
    test_dir = "../ft_whisper_results/l_ft_whisper_results"
    train_dir = "../whisper_l_train_data"
    data_name = "combined_transcripts_test.csv" if test else "combined_transcripts_train.csv"
    afri_data_name = "afri_transcripts_test.csv" if test else "afri_transcripts_train.csv"
    indic_data_name = "indic_transcripts_test.csv" if test else "indic_transcripts_train.csv"
    output_dir = "test_data" if test else "train_data"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = test_dir if test else train_dir
    afri_df = load_all_jsons(os.path.join(data_dir, "afri-5/*.json"))
    print("Afri Data Stats")
    print(afri_df.shape)

    indic_df = load_all_jsons(os.path.join(data_dir, "indic-5/*.json"))
    print("Indic Data Stats")
    print(indic_df.shape)

    afri_df.to_csv(os.path.join(output_dir, afri_data_name), index=False)
    indic_df.to_csv(os.path.join(output_dir, indic_data_name), index=False)

    # combined
    df = pd.concat([afri_df, indic_df])
    df.to_csv(os.path.join(output_dir, data_name), index=False)

    # metrics
    if test:
        df.dropna(inplace=True)
        df['base_wer'] = df.apply(lambda row: WERMetric().calculate([row['gold_transcript']], [row['whisper_l_ft']])[0],
                                  axis=1)
        # df['base_cer'] = df.apply(lambda row: CERMetric().calculate([row['gold_transcript']], [row['whisper_l_ft']]),
        #                           axis=1)
        df.to_csv(os.path.join(output_dir, data_name), index=False)

        afri_df.dropna(inplace=True)
        afri_df['base_wer'] = afri_df.apply(lambda row: WERMetric().calculate([row['gold_transcript']], [row['whisper_l_ft']])[0],
                                            axis=1)
        afri_df.to_csv(os.path.join(output_dir, afri_data_name), index=False)

        indic_df.dropna(inplace=True)
        indic_df['base_wer'] = indic_df.apply(
            lambda row: WERMetric().calculate([row['gold_transcript']], [row['whisper_l_ft']])[0],
            axis=1)
        indic_df.to_csv(os.path.join(output_dir, indic_data_name), index=False)

        print(f"Combined Wer score: {df['base_wer'].mean()}")
        print(f"Afri Wer score: {afri_df['base_wer'].mean()}")
        print(f"Indic Wer score: {indic_df['base_wer'].mean()}")

        # save wer stats
        wer_summary = pd.DataFrame({
            'dataset': ['combined', 'afri', 'indic'],
            'mean_wer': [
                df['base_wer'].mean(),
                afri_df['base_wer'].mean(),
                indic_df['base_wer'].mean()
            ]
        })
        wer_summary_path = os.path.join(output_dir, "wer_summary.csv")
        wer_summary.to_csv(wer_summary_path, index=False)

    print(df.head())
    print(df.shape)


def convert_to_training_json(test=False):
    config = "test" if test else "train"
    data_dir = f"{config}_data"

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

    for region in ["afri", "indic"]:
        filepath = os.path.join(data_dir, f"{region}_transcripts_{config}.csv")
        outdir = os.path.join(data_dir, region)
        os.makedirs(outdir, exist_ok=True)

        df = pd.read_csv(filepath)
        for lang in lang_dict.keys():
            lang_df = df[df["language"] == lang]
            if lang_df.empty:
                continue

            iso1 = lang.split("_")[0]
            iso3 = lang_dict[lang]
            records = [
                {"translation": {iso1: str(row["whisper_l_ft"]), iso3: str(row["gold_transcript"])}}
                for _, row in lang_df.iterrows()
                ]

            outpath = os.path.join(outdir, f"{lang}.json")
            with open(outpath, "w") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)


def main(args):
    # if args.data == "test":
    #     build_data(test=True)
    # elif args.data == "both":
    #     build_data(test=True)
    #     build_data(test=False)
    # else:
    #     build_data(test=False)

    if args.type == "json":
        convert_to_training_json(test=True)
        convert_to_training_json(test=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train", help="test, train or both")
    parser.add_argument("--type", type=str, default="json", help="json or csv")
    arguments = parser.parse_args()
    main(arguments)

