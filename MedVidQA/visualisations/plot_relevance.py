import pandas as pd
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from tqdm.auto import tqdm


def min_max_scaling(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min())


def extrapolate_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["end"] = df["start"] + df["duration"]
    df["center"] = (df["end"] + df["start"]) / 2
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_file",
        default="data/raw/MedVidQA/train.json",
        help="input json file containing start and end time of answers.",
    )
    parser.add_argument(
        "--results_file",
        default="data/processed/transcript_passage/query_samples.csv",
        help="results file in a csv format containing similarity score.",
    )
    parser.add_argument(
        "--plots_folder",
        default="data/plots/comparison/",
        help="folder where plots will be saved.",
    )
    parser.add_argument(
        "--model_name",
        default="bm25",
        help="name of the model that will be appended to the plot.",
    )

    args = parser.parse_args()

    with open(args.gold_file) as fp:
        gold_results = json.load(fp)

    df = pd.read_csv(args.results_file)
    df = extrapolate_scores(df=df)

    if not os.path.exists(args.plots_folder):
        os.makedirs(args.plots_folder)

    for query_id in tqdm(df["qid"].unique().tolist()):
        tmp_df = df[df["qid"] == query_id]

        tmp_df.loc["score"] = min_max_scaling(tmp_df["score"])

        gold_answer = gold_results[query_id - 1]

        # print(tmp_df[['qid',"docno","text", "query"]])
        # print(gold_answer)
        # print('\n')
        # print(tmp_df["query"].unique())
        # print(gold_answer['question'])

        tmp = (
            tmp_df.set_index("score")[["start", "end", "center"]]
            .unstack()
            .reset_index()
            .set_index(0)
        )

        plt.plot(figsize=(15, 15))

        sns.lineplot(data=tmp["score"], ci=0.7)
        x1, y1 = [
            gold_answer["answer_start_second"],
            gold_answer["answer_end_second"],
        ], [0, 0]
        plt.plot(x1, y1, marker="x", color="red")
        plt.savefig(
            f"{args.plots_folder}/{gold_answer['sample_id']}_line_{args.model_name}.png",
            dpi=600,
        )
        plt.cla()

        sns.displot(data=tmp_df, x="center", y="score", kind="kde", rug=True, fill=True)
        plt.plot(x1, y1, marker="x", color="red")
        plt.savefig(
            f"{args.plots_folder}/{gold_answer['sample_id']}_kde_{args.model_name}.png",
            dpi=600,
        )
        plt.cla()
