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

# model = SentenceTransformer("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
# model = SentenceTransformer("bert-base-uncased")
model = SentenceTransformer("sentence-transformers/allenai-specter")
model = model.to(device)


def similarity_score(query: str, documents: list[str]):
    query_encoded: np.array = model.encode([query])
    documents_encoded: np.array = model.encode(documents)

    return cosine_similarity(query_encoded, documents_encoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript_data_path", default="data/interim/transcripts_merged/"
    )
    parser.add_argument("--question_data_path", default="data/raw/MedVidQA/")
    parser.add_argument(
        "--output_data_path", default="data/processed/predictions/passage_similarity/"
    )
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--model", default="specter-2")

    args = parser.parse_args()

    with open(f"{args.question_data_path}/{args.dataset}.json") as fp:
        videos = json.load(fp)

    with open(f"{args.transcript_data_path}/test_2.json") as fp:
        transcripts = json.load(fp)

    if not os.path.exists(f"{args.output_data_path}/{args.dataset}"):
        os.makedirs(f"{args.output_data_path}/{args.dataset}")

    out_df = pd.DataFrame()
    for video in tqdm(videos):
        transcript = [x for x in transcripts if x["video_id"] == video["video_id"]][0]

        query = video["question"]
        documents = [line["text"] for line in transcript["transcript"]]
        sim_scores = similarity_score(query=query, documents=documents)

        df = pd.DataFrame.from_dict(transcript["transcript"])
        df["score"] = sim_scores[0]
        df["score"] = min_max_scaling(df["score"])

        df["qid"] = video["sample_id"]
        df["model"] = args.model

        out_df = pd.concat([out_df, df])

        out_df.to_csv(
            f"{args.output_data_path}/{args.dataset}/{args.model}_{args.dataset}.csv"
        )
