import argparse
import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

models_list = [
    "msmarco_roberta",
    "nli_mpnet",
    "BM25",
    "DirichletLM",
    # "specter",
    # "multiqa_minilm",
    # "bert",
    # "bluebert",
    # "Tf",
    # "In_expB2"
]


def plot_prediction(y_true, y_pred, label: str, c: str, marker: str, title: str):
    # Plot start time
    plt.figure()
    s = 50
    a = 0.4
    plt.scatter(
        y_true,
        y_pred,
        edgecolor="k",
        c=c,
        s=s,
        marker=marker,
        alpha=a,
        label=label,
    )
    plt.plot(range(0, 3), range(0, 3), color="red")
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel("True")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.legend()

    return plt


def load_data(gold_file, predictions_file):
    df = pd.read_csv(predictions_file)

    with open(gold_file) as fp:
        gold_results = json.load(fp)

    gold_results = [x for x in gold_results if x["sample_id"] in df["qid"].unique()]

    df = df[df["model"].isin(models_list)]
    df = df[df["sampling"].isin(["max", "median"])]

    try:
        X = pd.pivot_table(df, values='score', index=['qid'],
                           columns=['sample_n', 'model', 'sampling'])
    except:
        print("can't pivot larger data")
        X = pd.pivot_table(df, values="score", index=["qid"], columns=["model", "order_n"])
    X = X.fillna(0)
    X = X.values

    # append video length
    if len(X) == len(gold_results):
        X = np.c_[X, np.array([x["video_length"] for x in gold_results])]

    query_dict = {index_i: x["sample_id"] for index_i, x in enumerate(gold_results)}

    try:
        y = [
            (
                x["answer_start_second"] / x["video_length"],
                (x["answer_end_second"] - x["answer_start_second"]) / x["video_length"],
            )
            for x in gold_results
        ]
        y = np.array(y)
    except KeyError:
        y = None

    return X, y, query_dict, gold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_files_folder",
        default="data/raw/MedVidQA/",
        help="input json file containing start and end time of answers.",
    )
    parser.add_argument(
        "--prediction_folder",
        default="data/processed/predictions/time_normalised/",
        help="results file in a csv format containing similarity score.",
    )
    parser.add_argument(
        "--sampling_type",
        default="mean",
        help="sampling type - min, max, mean. supported by time_normalisation script.",
    )
    parser.add_argument(
        "--submission_data_path",
        default="submission_4-models_median-max",
    )
    args = parser.parse_args()

    X_final, y_final, final_dict, gold_final = load_data(
        gold_file=f"{args.gold_files_folder}/final.json",
        predictions_file=f"{args.prediction_folder}/{args.sampling_type}/final.csv",
    )
    X_train, y_train, _, _ = load_data(
        gold_file=f"{args.gold_files_folder}/train.json",
        predictions_file=f"{args.prediction_folder}/{args.sampling_type}/train.csv",
    )
    X_val, y_val, _, _ = load_data(
        gold_file=f"{args.gold_files_folder}/val.json",
        predictions_file=f"{args.prediction_folder}/{args.sampling_type}/val.csv",
    )
    X_test, y_test, test_dict, gold_test = load_data(
        gold_file=f"{args.gold_files_folder}/test.json",
        predictions_file=f"{args.prediction_folder}/{args.sampling_type}/test.csv",
    )
    X_train = np.concatenate((np.concatenate((X_train, X_val), axis=0), X_test), axis=0)
    y_train = np.concatenate((np.concatenate((y_train, y_val), axis=0), y_test), axis=0)
    print("data loaded")

    max_depth = 10
    n_estimators = 40
    regr_multirf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    )
    regr_multirf.fit(X_train, y_train)

    regr_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    regr_rf.fit(X_train, y_train)

    # Predict on new data
    y_multirf = regr_multirf.predict(X_final)
    y_rf = regr_rf.predict(X_final)

    # predict on train to check overfit
    y_multirf_train = regr_multirf.predict(X_train)

    DO_PLOTS = False
    if DO_PLOTS:
        # plot predictions on train data
        plt = plot_prediction(
            y_true=y_train[:, 0],
            y_pred=y_multirf_train[:, 0],
            label="Multi RF score=%.2f" % regr_multirf.score(X_train, y_train),
            c="navy",
            marker="s",
            title="Comparing multi-output RF for\nstart time prediction on train with %s sampling"
            % (args.sampling_type),
        )
        plt.show()

        plt = plot_prediction(
            y_true=y_train[:, 1],
            y_pred=y_multirf_train[:, 1],
            label="Multi RF score=%.2f" % regr_multirf.score(X_train, y_train),
            c="navy",
            marker="s",
            title="Comparing multi-output RF for\nduration prediction on train with %s sampling"
            % (args.sampling_type),
        )
        plt.show()

        # plot predictions on test data
        plt = plot_prediction(
            y_true=y_test[:, 0],
            y_pred=y_multirf[:, 0],
            label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test),
            c="navy",
            marker="s",
            title="Comparing multi-output RF for\nstart time prediction on test with %s sampling"
            % (args.sampling_type),
        )
        plt.show()

        plt = plot_prediction(
            y_true=y_test[:, 1],
            y_pred=y_multirf[:, 1],
            label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test),
            c="navy",
            marker="s",
            title="Comparing multi-output RF for\nduration prediction on test with %s sampling"
            % (args.sampling_type),
        )
        plt.show()
        plt.show()

    # create submission
    if not os.path.exists(f"{args.prediction_folder}/{args.sampling_type}/{args.submission_data_path}"):
        os.makedirs(f"{args.prediction_folder}/{args.sampling_type}/{args.submission_data_path}")

    submission = []
    for index_i, sample_id in final_dict.items():
        video = [x for x in gold_final if x["sample_id"] == sample_id][0]

        start_time = y_multirf[index_i, 0] * video["video_length"]
        duration = y_multirf[index_i, 1] * video["video_length"]
        end_time = start_time + duration

        submission.append(
            {
                "sample_id": sample_id,
                "answer_start_second": start_time,
                "answer_end_second": end_time,
            }
        )

    with open(
        f"{args.prediction_folder}/{args.sampling_type}/{args.submission_data_path}/test.json",
        "w",
    ) as fp:
        json.dump(submission, fp, indent=2)
