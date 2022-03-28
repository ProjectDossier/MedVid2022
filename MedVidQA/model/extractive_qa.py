import argparse
import json
import os

from tqdm import tqdm
from transformers import pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_data_path", default="data/interim/transcripts/")
    parser.add_argument("--question_data_path", default="data/raw/MedVidQA/")
    parser.add_argument(
        "--output_data_path", default="data/processed/text_predictions/"
    )
    parser.add_argument("--filename", default="train.json")
    parser.add_argument("--model", default="distilbert_squad")

    args = parser.parse_args()

    with open(f"{args.question_data_path}/{args.filename}") as fp:
        videos = json.load(fp)

    with open(f"{args.transcript_data_path}/{args.filename}") as fp:
        transcripts = json.load(fp)

    # merge transcript text
    for transcript in transcripts:
        merged = ""
        for line in transcript["transcript"]:
            try:
                merged += line["text"] + " "
            except TypeError:
                merged = "empty transcript"
        transcript["merged"] = merged.strip()

    question_answering = pipeline(
        "question-answering", model="distilbert-base-uncased-distilled-squad"
    )

    for video in tqdm(videos):
        transcript = [x for x in transcripts if x["video_id"] == video["video_id"]][0]
        result = question_answering(
            question=video["question"], context=transcript["merged"]
        )

        video["prediction"] = result

    if not os.path.exists(f"{args.output_data_path}/{args.model}/"):
        os.makedirs(f"{args.output_data_path}/{args.model}/")

    with open(f"{args.output_data_path}/{args.model}/{args.filename}", "w") as fp:
        json.dump(videos, fp, indent=2)
