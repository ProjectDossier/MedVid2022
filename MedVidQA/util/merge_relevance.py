import argparse
import os

import pandas as pd
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder", default="data/processed/predictions/normalised_predictions/"
    )
    parser.add_argument(
        "--output_folder",
        default="data/processed/predictions/merged_predictions/",
    )
    # parser.add_argument("--dataset", default="test")

    args = parser.parse_args()

    data_types: list[str] = [
        x
        for x in os.listdir(f"{args.input_folder}")
        if os.path.isdir(f"{args.input_folder}/{x}")
    ]
    for data_type in tqdm(data_types):

        csv_files = [
            x
            for x in os.listdir(f"{args.input_folder}/{data_type}")
            if x.endswith(".csv")
        ]

        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

        out_df = pd.DataFrame()
        for csv_file in csv_files:
            df = pd.read_csv(f"{args.input_folder}/{data_type}/{csv_file}")
            out_df = pd.concat([out_df, df])

        # FIXME in statistical models - remove in case of empty OCR...
        out_df = out_df[out_df["start"] != "Empty"]
        out_df['start'] = out_df['start'].astype(float)
        out_df['duration'] = out_df['duration'].astype(float)
        out_df['score'] = out_df['score'].astype(float)
        out_df = out_df.sort_values(["qid", "start", "duration", "score"])
        out_df = out_df.drop_duplicates()
        out_df.to_csv(f"{args.output_folder}/{data_type}.csv", index=False)
