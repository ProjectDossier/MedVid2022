import json
import argparse
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity


device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    parser.add_argument("--filename", default="test.json")
    parser.add_argument("--model", default="specter-3")

    args = parser.parse_args()

    with open(f"{args.question_data_path}/{args.filename}") as fp:
        videos = json.load(fp)

    with open(f"{args.transcript_data_path}/test_3.json") as fp:
        transcripts = json.load(fp)

    out_df = pd.DataFrame()
    for video in tqdm(videos):
        transcript = [x for x in transcripts if x["video_id"] == video["video_id"]][0]

        query = video["question"]
        documents = [line["text"] for line in transcript["transcript"]]
        sim_scores = similarity_score(query=query, documents=documents)

        df = pd.DataFrame.from_dict(transcript["transcript"])
        df["score"] = sim_scores[0]
        df["qid"] = video["sample_id"]  # FIXME
        # video["prediction"] = sim_scores
        out_df = out_df.append(df)

        out_df.to_csv(
            f"{args.output_data_path}/{args.model}_{args.filename.split('.')[0]}.csv"
        )
