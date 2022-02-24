import pandas as pd
import argparse
import os
from tqdm.auto import tqdm

from MedVidQA.util.data_util import min_max_scaling

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder", default="data/processed/predictions/passage_similarity/"
    )
    parser.add_argument(
        "--output_folder", default="data/processed/predictions/normalised_predictions/"
    )
    parser.add_argument(
        "--output_name",
        default="normalised",
    )
    args = parser.parse_args()

    data_types: list[str] = [
        x for x in os.listdir(f"{args.input_folder}") if os.path.isdir(f"{args.input_folder}/{x}")
    ]
    for data_type in tqdm(data_types):

        if not os.path.exists(f"{args.output_folder}/{data_type}"):
            os.makedirs(f"{args.output_folder}/{data_type}")

        models = [
            x
            for x in os.listdir(f"{args.input_folder}/{data_type}")
            if os.path.isdir(f"{args.input_folder}/{data_type}/{x}")
        ]
        for model in models:
            input_folder = f"{args.input_folder}/{data_type}/{model}/"
            output_file = f"{args.output_folder}/{data_type}/{model}_{args.output_name}.csv"

            csv_files = [x for x in os.listdir(f"{input_folder}/") if x.endswith(".csv")]

            out_df = pd.DataFrame()
            for csv_file in csv_files:
                df = pd.read_csv(f"{input_folder}/{csv_file}", index_col=0)
                out_df = pd.concat([out_df, df])

            # normlise score by qid
            for qid in out_df['qid'].unique().tolist():
                out_df.loc[out_df['qid'] == qid, 'score'] = min_max_scaling(out_df.loc[out_df['qid'] == qid, 'score'])

            out_df = out_df.sort_values(["qid","start","duration","score"])
            out_df = out_df.drop("text", axis=1)
            out_df.to_csv(output_file, index=False)



