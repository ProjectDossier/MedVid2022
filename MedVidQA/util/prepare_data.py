"""script for merging multiple subtitle lines into one larger text.
It simply concatenates text and take min and max starting times from merged lines."""
import json
import argparse
from tqdm.auto import tqdm


def merge_transcripts(
    transcript: list[
        dict[
            str,
        ]
    ],
    every_n: int = 2,
) -> list[dict[str,]]:
    out_dict = []
    for index_i, line in enumerate(transcript):
        try:  # if transcript doesn't exist
            line["start"]
        except TypeError:
            return transcript

        min_time = line["start"]
        max_time = line["start"] + line["duration"]
        out_string = line["text"]
        for offset in range(1, every_n):
            if index_i + offset >= len(transcript):
                break
            out_string += " " + transcript[index_i + offset]["text"]
            if (
                max_time
                < transcript[index_i + offset]["start"]
                + transcript[index_i + offset]["duration"]
            ):
                max_time = (
                    transcript[index_i + offset]["start"]
                    + transcript[index_i + offset]["duration"]
                )

        out_dict.append(
            {"text": out_string, "start": min_time, "duration": max_time - min_time}
        )
    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", default="data/interim/transcripts/")
    parser.add_argument(
        "--output_data_path", default="data/interim/transcripts_merged/"
    )
    parser.add_argument("--filename", default="train.json")

    args = parser.parse_args()

    for merge_n_lines in tqdm(range(2, 5)):
        with open(f"{args.input_data_path}/{args.filename}") as fp:
            transcripts = json.load(fp)

        for video in transcripts:
            video["transcript"] = merge_transcripts(
                transcript=video["transcript"], every_n=merge_n_lines
            )

        outfile = f"{args.output_data_path}/{args.filename}_{merge_n_lines}.json"
        with open(
            outfile,
            "w",
        ) as fp:
            json.dump(transcripts, fp, indent=2)
