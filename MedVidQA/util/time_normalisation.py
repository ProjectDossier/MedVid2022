import pandas as pd
import pandas as pd
import os
import argparse
from tqdm.auto import tqdm
import json


def normalise_time(data: pd.DataFrame, video_duration:float) -> pd.DataFrame:
    df = data.copy()

    df["start"] = df["start"] / video_duration
    df["duration"] = df["duration"] / video_duration
    return df


def inverse_normalise_time(data: pd.DataFrame, video_duration:float) -> pd.DataFrame:
    df = data.copy()

    df["start"] = df["start"] * video_duration
    df["duration"] = df["duration"] * video_duration
    return df


def sample_results(data:pd.DataFrame, n_samples:int=100, type="minmax") ->pd.DataFrame:
    # df = data.copy()
    out_df = pd.DataFrame()
    if type == "minmax":
        for model in data['model'].unique():
            df = data[data['model'] == model].copy()

            for n in range(n_samples):
                lower_bound = ((n -1 )/ n_samples + n / n_samples) / 2
                upper_bound = (n/ n_samples + (n + 1) / n_samples) / 2

                tmp_df = df[(lower_bound <= df['start']) & (df['start'] <= upper_bound)].copy()
                if len(tmp_df) == 0:
                    tmp_df = df.iloc[:2].copy()
                    tmp_df["start"] = lower_bound
                    tmp_df["duration"] = upper_bound - lower_bound
                    tmp_df["score"] = 0

                tmp_df["order_n"] = n
                # print(tmp_df["score"].idxmax())
                # print(tmp_df)
                # dd = tmp_df.loc[tmp_df["score"].idxmax()]
                out_df = pd.concat([out_df, tmp_df.loc[tmp_df["score"].idxmax()]], axis=1)
                out_df = pd.concat([out_df, tmp_df.loc[tmp_df["score"].idxmin()]], axis=1)
    elif type == "mean":
        pass
    print(out_df.shape)
    return out_df.T

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder", default="data/processed/predictions/merged_predictions/"
    )
    parser.add_argument(
        "--gold_folder", default="data/raw/MedVidQA/"
    )
    parser.add_argument(
        "--output_folder",
        default="data/processed/predictions/time_normalised/",
    )

    dataset = "test"

    args = parser.parse_args()

    with open(f"{args.gold_folder}/{dataset}.json") as fp:
        gold_results = json.load(fp)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    df = pd.read_csv(f"{args.input_folder}/{dataset}.csv")
    df["score"] = df["score"].fillna(0)

    out_df = pd.DataFrame()

    stats_df = pd.DataFrame()

    for query_id in tqdm(df["qid"].unique().tolist()):
        tmp_df = df[df["qid"] == query_id].copy()

        gold_answer = [x for x in gold_results if x["sample_id"] == query_id][0]

        video_length = gold_answer["video_length"]

        stats_df = pd.concat([stats_df, pd.DataFrame({
            "len_items":[len(tmp_df)],
            "unique_starts":[len(tmp_df["start"].unique())],
            "video_len":[video_length],
            "video_id":[gold_answer["video_id"]]
        })])

        result_df = normalise_time(data=df[df["qid"] == query_id].copy(), video_duration=video_length)
        result_df = sample_results(data=result_df, n_samples=50, type="minmax")
        result_df = inverse_normalise_time(data=result_df, video_duration=video_length)

        out_df = pd.concat([out_df, result_df])

    out_df.to_csv(f"{args.output_folder}/{dataset}.csv")
    stats_df.to_csv(f"{args.output_folder}/{dataset}_stats.csv")
