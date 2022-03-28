import argparse
import json
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_files_folder",
        default="data/raw/MedVidQA/",
        help="input json file containing start and end time of answers.",
    )
    parser.add_argument(
        "--prediction_folder",
        default="data/processed/peak_prediction/",
        help="results file in a csv format containing similarity score.",
    )
    parser.add_argument(
        "--model_name",
        default="PD_4",
        help="Name of the peak detection model, either PD_4 or PD_2.",
    )
    parser.add_argument(
        "--submission_data_path",
        default="submission_4_models",
    )
    parser.add_argument("--dataset", default="test")
    args = parser.parse_args()

    out_folder = f"{args.prediction_folder}/{args.submission_data_path}"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(f"{args.gold_files_folder}/{args.dataset}.json") as fp:
        gold_results = json.load(fp)

    df = pd.read_csv(
        f"data/processed/predictions/merged_predictions/{args.dataset}.csv"
    )

    df["center"] = (df["duration"] + 2 * df["start"]) / 2

    if args.model_name == "PD_4":
        subset_df = df[
            df["model"].isin(["msmarco_roberta", "DirichletLM", "BM25", "nli_mpnet"])
        ]
    elif args.model_name == "PD_2":
        subset_df = df[df["model"].isin(["msmarco_roberta", "DirichletLM"])]
    else:
        raise ValueError("args.model_name can be either PD_4 or PD_2")

    peaks_df = (
        subset_df.groupby(["qid", "center"])["score"]
        .mean()
        .iloc[
            subset_df.groupby(["qid", "center"])["score"]
            .mean()
            .reset_index()
            .groupby("qid")["score"]
            .idxmax()
        ]
    )
    peaks_df = peaks_df.reset_index()

    BETA_1 = -6
    BETA_2 = 62

    submission = []
    for sample_id in peaks_df["qid"].tolist():
        video = [x for x in gold_results if x["sample_id"] == sample_id][0]

        start_time = (
            peaks_df[peaks_df["qid"] == sample_id]["center"].tolist()[0] + BETA_1
        )

        end_time = start_time + BETA_2

        submission.append(
            {
                "sample_id": sample_id,
                "answer_start_second": start_time,
                "answer_end_second": end_time,
            }
        )

    with open(
        f"{out_folder}/{args.dataset}.json",
        "w",
    ) as fp:
        json.dump(submission, fp, indent=2)
