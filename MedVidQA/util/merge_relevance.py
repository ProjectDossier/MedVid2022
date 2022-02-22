import pandas as pd
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder", default="data/processed/predictions/passage_similarity/"
    )
    parser.add_argument(
        "--output_folder",
        default="data/processed/predictions/passage_similarity/merged/",
    )
    parser.add_argument("--dataset", default="test")

    args = parser.parse_args()

    csv_files = [
        x
        for x in os.listdir(f"{args.input_folder}/{args.dataset}")
        if x.endswith(".csv")
    ]

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    out_df = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(f"{args.input_folder}/{args.dataset}/{csv_file}")
        out_df = pd.concat([out_df, df])

    out_df.to_csv(f"{args.output_folder}/{args.dataset}.csv")
