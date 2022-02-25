import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import argparse
import pandas as pd
import json


def plot_prediction(y_true, y_pred, label:str, c:str, marker:str, title:str):
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


def load_data(gold_file, predictions):

    with open(gold_file) as fp:
        gold_results = json.load(fp)

    df = pd.read_csv(predictions)

    print(df['qid'].unique())
    # print(len(df['qid'].unique()))
    # print([x for x in range(1,2711) if x not in df['qid'].unique()])


    df = df[df['model'].isin(["msmarco_roberta", "bert", "bluebert", "nli_mpnet", "specter", "multiqa_minilm"])]

    X = pd.pivot_table(df, values='score', index=['qid'],
                        columns=['model', 'order_n'])

    X = X.fillna(0)
    X = X.values
    # X = np.c_[X, np.array([x['video_length'] for x in gold_results])]

    query_dict = {
        index_i:x['sample_id'] for index_i, x in enumerate(gold_results)
    }
    print(query_dict)
    y = [(x['answer_start_second']/x['video_length'], (x['answer_end_second']-x['answer_start_second'])/x['video_length']) for x in gold_results]
    y = np.array(y)

    # FIXME for train
    if len(y) > 600:
        y = np.delete(y, (674), axis=0)

    # print(y[674])
    # y = np.c_(y[:675], y[676:])
    print(np.argwhere(np.isnan(y)))
    print(np.argwhere(np.isnan(X)))

    print(X.shape)
    print(y.shape)

    return X, y, query_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_file",
        default="data/raw/MedVidQA/test.json",
        help="input json file containing start and end time of answers.",
    )
    parser.add_argument(
        "--results_file",
        default="data/processed/predictions/time_normalised/test.csv",
        help="results file in a csv format containing similarity score.",
    )
    parser.add_argument(
        "--plots_folder",
        default="data/plots/comparison/",
        help="folder where plots will be saved.",
    )
    parser.add_argument(
        "--model_name",
        default="time_normalised",
        help="name of the model that will be appended to the plot.",
    )
    parser.add_argument(
        "--submission_data_path",
        default="data/processed/predictions/time_normalised/submission/",
    )

    args = parser.parse_args()

    with open(args.gold_file) as fp:
        gold_results = json.load(fp)

    df = pd.read_csv(args.results_file)

    print(df['qid'].unique())
    # print(len(df['qid'].unique()))
    # print([x for x in range(1,2711) if x not in df['qid'].unique()])


    # df = df[df['model'].isin(["msmarco_roberta", "bert", "bluebert", "nli_mpnet", "specter", "multiqa_minilm"])]

    X = pd.pivot_table(df, values='score', index=['qid'],
                        columns=['model', 'order_n'])

    X = X.fillna(0)
    X = X.values
    # X = np.c_[X, np.array([x['video_length'] for x in gold_results])]

    query_dict = {
        index_i:x['sample_id'] for index_i, x in enumerate(gold_results)
    }
    print(query_dict)
    y = [(x['answer_start_second']/x['video_length'], x['answer_end_second']/x['video_length']) for x in gold_results]
    y = np.array(y)

    # FIXME for train
    # y = np.delete(y, (674), axis=0)

    # print(y[674])
    # y = np.c_(y[:675], y[676:])
    print(np.argwhere(np.isnan(y)))
    print(np.argwhere(np.isnan(X)))

    print(X.shape)
    print(y.shape)

    X_train, y_train, train_dict = load_data(gold_file="data/raw/MedVidQA/train.json",
                                             predictions="data/processed/predictions/time_normalised/train.csv")
    X_test, y_test, test_dict = load_data(gold_file="data/raw/MedVidQA/test.json",
                                          predictions="data/processed/predictions/time_normalised/test.csv")

    # Create a random dataset
    # rng = np.random.RandomState(1)
    # X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
    # y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    # y += 0.5 - rng.rand(*y.shape)
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, train_size=0.8, random_state=4
    # )

    max_depth = 20
    regr_multirf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=3, max_depth=max_depth, random_state=0)
    )
    regr_multirf.fit(X_train, y_train)

    regr_rf = RandomForestRegressor(n_estimators=2, max_depth=max_depth, random_state=2)
    regr_rf.fit(X_train, y_train)

    # Predict on new data
    y_multirf = regr_multirf.predict(X_test)
    y_rf = regr_rf.predict(X_test)


    # plot predictions on train data
    y_multirf_train = regr_multirf.predict(X_train)
    plt = plot_prediction(y_true=y_train[:, 0], y_pred=y_multirf_train[:, 0],
                    label="Multi RF score=%.2f" % regr_multirf.score(X_train, y_train),
                    c="navy", marker="s",
                    title="Comparing multi-output RF \nfor start time prediction on train")
    plt.show()

    plt = plot_prediction(y_true=y_train[:, 1], y_pred=y_multirf_train[:, 1],
                    label="Multi RF score=%.2f" % regr_multirf.score(X_train, y_train),
                    c="navy", marker="s",
                    title="Comparing multi-output RF \nfor end time prediction on train")
    plt.show()

    # plot predictions on test data
    plt = plot_prediction(y_true=y_test[:, 0], y_pred=y_multirf[:, 0],
                    label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test),
                    c="navy", marker="s",
                    title="Comparing multi-output RF \nfor start time prediction on test")
    plt.show()

    plt = plot_prediction(y_true=y_test[:, 1], y_pred=y_multirf[:, 1],
                    label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test),
                    c="navy", marker="s",
                    title="Comparing multi-output RF \nfor end time prediction on test")
    plt.show()
    plt.show()


    # create submission
    submission = []
    with open("data/raw/MedVidQA/test.json") as fp:
        gold_results = json.load(fp)

    for index_i, sample_id in test_dict.items():
        video = [x for x in gold_results if x["sample_id"] == sample_id][0]
    # for video in videos:
        start_time = y_multirf[index_i,0]*video["video_length"]
        duration = y_multirf[index_i, 1] * video["video_length"]
        end_time = start_time + duration
        # end_time = y_multirf[index_i, 1] * video["video_length"]
        # if start_time > y_multirf[index_i,1]*video["video_length"]:
        #     start_time = y_multirf[index_i,1]*video["video_length"]
        #     end_time = y_multirf[index_i,0]*video["video_length"]
        submission.append(
            {
                "sample_id": sample_id,
                "answer_start_second": start_time,
                "answer_end_second": end_time,
            }
        )

    if not os.path.exists(args.submission_data_path):
        os.makedirs(args.submission_data_path)

    with open(
        f"{args.submission_data_path}/test.json",
        "w",
    ) as fp:
        json.dump(submission, fp, indent=2)



    # for row in y_multirf:
    #
    # y_pred_time = [(x['answer_start_second']/x['video_length'], x['answer_end_second']/x['video_length']) for x in y_multirf]



