import json
import argparse
import os.path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from MedVidQA.util.data_util import min_max_scaling

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"using {device}")


def similarity_score(model: SentenceTransformer, query: str, documents: list[str]):
    query_encoded: np.array = model.encode([query])
    documents_encoded: np.array = model.encode(documents)

    return cosine_similarity(query_encoded, documents_encoded)


def select_model(model_name: str) -> SentenceTransformer:
    if model_name == "bluebert":
        model = SentenceTransformer(
            "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
        )
    elif model_name == "bert":
        model = SentenceTransformer("bert-base-uncased")
    elif model_name == "specter":
        model = SentenceTransformer("sentence-transformers/allenai-specter")
    elif model_name == "multiqa_minilm":
        model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    elif model_name == "msmarco_roberta":
        model = SentenceTransformer(
            "sentence-transformers/msmarco-distilroberta-base-v2"
        )
    elif model_name == "nli_mpnet":
        model = SentenceTransformer("sentence-transformers/nli-mpnet-base-v2")
    else:
        model = SentenceTransformer("bert-base-uncased")
        print("WARN: using default BERT model")
    model = model.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_data_path", default="data/interim/transcripts/")
    parser.add_argument(
        "--transcripts_file",
        default="",
        help="if empty it will take every json in the folder",
    )
    parser.add_argument("--question_data_path", default="data/raw/MedVidQA/")
    parser.add_argument(
        "--output_data_path", default="data/processed/predictions/passage_similarity/"
    )
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--model", default="specter", type=str)

    args = parser.parse_args()

    if not os.path.exists(f"{args.output_data_path}/{args.dataset}/{args.model}"):
        os.makedirs(f"{args.output_data_path}/{args.dataset}/{args.model}")

    model: SentenceTransformer = select_model(model_name=args.model)

    with open(f"{args.question_data_path}/{args.dataset}.json") as fp:
        videos = json.load(fp)

    input_transcripts_dict: dict = {}
    if args.transcripts_file:
        with open(f"{args.transcript_data_path}/{args.transcripts_file}") as fp:
            input_transcripts_dict[args.transcripts_file] = json.load(fp)
    else:
        transcripts_files = [
            x for x in os.listdir(args.transcript_data_path) if x.endswith(".json")
        ]
        for transcripts_file in transcripts_files:
            with open(f"{args.transcript_data_path}/{transcripts_file}") as fp:
                input_transcripts_dict[transcripts_file] = json.load(fp)

    print(f"Loaded {len(input_transcripts_dict)} transcript datasets")

    for transcripts_file, transcripts in input_transcripts_dict.items():
        print(f"Processing file: {transcripts_file}")
        input_feature = transcripts_file.split(".")[0]

        out_df = pd.DataFrame()
        for video in tqdm(videos):
            transcript = [x for x in transcripts if x["video_id"] == video["video_id"]]
            if len(transcript) > 0:
                # we need to take the first element from the transcript list of all
                # transcripts with the same video_id
                transcript = transcript[0]["transcript"]

                # sometimes it is possible that we have an empty transcript record
                if len(transcript) == 0:
                    continue
            else:
                continue

            query = video["question"]
            try:
                documents = [line["text"] for line in transcript]
            except TypeError:  # FIXME - change transcript from youtube videos to return list in case of empty transcript
                continue
            sim_scores = similarity_score(model=model, query=query, documents=documents)

            df = pd.DataFrame.from_dict(transcript)
            df["score"] = sim_scores[0]
            # df["score"] = min_max_scaling(df["score"])

            df["qid"] = video["sample_id"]
            df["model"] = args.model
            df["input_feature"] = input_feature

            out_df = pd.concat([out_df, df])

            out_df.to_csv(
                f"{args.output_data_path}/{args.dataset}/{args.model}/{input_feature}.csv"
            )
