import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from MedVidQA.util.data_util import min_max_scaling


def extrapolate_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["end"] = df["start"] + df["duration"]
    df["center"] = (df["end"] + df["start"]) / 2
    return df


def plot_results_line(
    data: pd.DataFrame,
    answer_start_second: float,
    answer_end_second: float,
    filename: str,
    hue: str = None,
    x_column: str = None,
    y_column: str = None,
):
    x1, y1 = [
        answer_start_second,
        answer_end_second,
    ], [0, 0]

    plt.plot(figsize=(15, 15))
    if hue and x_column and y_column:
        sns.lineplot(data=data, x=x_column, y=y_column, hue=hue, size=0.15)
    else:
        sns.lineplot(data=data)
    plt.plot(x1, y1, marker="x", color="red")
    plt.ylabel("similarity scores")
    plt.xlabel("time [s]")
    plt.savefig(
        filename,
        dpi=300,
    )
    plt.cla()


def plot_results_kde(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    answer_start_second: float,
    answer_end_second: float,
    filename: str,
):
    x1, y1 = [
        answer_start_second,
        answer_end_second,
    ], [0, 0]

    plt.plot(figsize=(15, 15))
    sns.displot(data=data, x=x_column, y=y_column, kind="kde", rug=True, fill=True)
    plt.plot(x1, y1, marker="x", color="red")
    plt.savefig(
        filename,
        dpi=600,
    )
    plt.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_file",
        default="data/raw/MedVidQA/test.json",
        help="input json file containing start and end time of answers.",
    )
    parser.add_argument(
        "--results_file",
        default="data/processed/predictions/merged_predictions/test.csv",
        help="results file in a csv format containing similarity score.",
    )
    parser.add_argument(
        "--plots_folder",
        default="data/plots/comparison/",
        help="folder where plots will be saved.",
    )
    parser.add_argument(
        "--model_name",
        default="test",
        help="name of the model that will be appended to the plot.",
    )
    MINMAX_SCALE = False
    PLOT_KDE = False

    args = parser.parse_args()

    with open(args.gold_file) as fp:
        gold_results = json.load(fp)

    df = pd.read_csv(args.results_file)
    df = extrapolate_scores(df=df)

    plots_outfolder = f"{args.plots_folder}/{args.model_name}/"
    if not os.path.exists(plots_outfolder):
        os.makedirs(plots_outfolder)

    for query_id in tqdm(df["qid"].unique().tolist()):
        tmp_df = df[df["qid"] == query_id].copy()

        if MINMAX_SCALE:
            tmp_df["score"] = min_max_scaling(tmp_df["score"])

        gold_answer = [x for x in gold_results if x["sample_id"] == query_id][0]

        tmp = (
            tmp_df.set_index("score")[["start", "end", "center"]]
            .unstack()
            .reset_index()
            .set_index(0)
        )

        plot_results_line(
            data=tmp["score"],
            answer_start_second=gold_answer["answer_start_second"],
            answer_end_second=gold_answer["answer_end_second"],
            filename=f"{plots_outfolder}/{gold_answer['sample_id']}_line_{args.model_name}.png",
        )
        tmp_df.loc[tmp_df["input_feature"] == "test", "input_feature"] = "transcript-1"
        tmp_df.loc[tmp_df["input_feature"] == "test_2", "input_feature"] = "transcript-2"
        tmp_df.loc[tmp_df["input_feature"] == "test_3", "input_feature"] = "transcript-3"
        tmp_df.loc[tmp_df["input_feature"] == "test_4", "input_feature"] = "transcript-4"

        # input_features
        plot_results_line(
            data=tmp_df,
            x_column="center",
            y_column="score",
            hue="input_feature",
            answer_start_second=gold_answer["answer_start_second"],
            answer_end_second=gold_answer["answer_end_second"],
            filename=f"{plots_outfolder}/{gold_answer['sample_id']}_document_representations.png",
        )

        plot_results_line(
            data=tmp_df[
                tmp_df["model"].isin(
                    [
                        "msmarco_roberta",
                        "BM25",
                        "DirichletLM",
                        "nli_mpnet",
                    ]
                )
            ],
            x_column="center",
            y_column="score",
            hue="model",
            answer_start_second=gold_answer["answer_start_second"],
            answer_end_second=gold_answer["answer_end_second"],
            filename=f"{plots_outfolder}/{gold_answer['sample_id']}_models.png",
        )

        if PLOT_KDE:
            plot_results_kde(
                data=tmp_df,
                x_column="center",
                y_column="score",
                answer_start_second=1,
                answer_end_second=2,
                filename=f"{plots_outfolder}/{gold_answer['sample_id']}_kde_{args.model_name}.png",
            )
