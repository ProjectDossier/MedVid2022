import json
import argparse
from tqdm.auto import tqdm
import os


def text_to_time_chars(
    char_position_start: int,
    char_position_end: int,
    transcript: list[dict],
    merge_type: str = "strict",
):
    """converts prediction written as a start and end characters

    :param char_position_start:
    :param char_position_end:
    :param transcript:
    :param merge_type:
    """
    start_time: float = 0
    end_time: float = 0

    if merge_type == "strict":
        current_char_position = 0
        start_time_found = False
        for line in transcript:
            if (
                not start_time_found
                and len(line["text"]) + current_char_position > char_position_start
            ):
                start_time = line["start"]
            if len(line["text"]) + current_char_position > char_position_end:
                end_time = line["start"] + line["duration"]
            current_char_position += len(line["text"]) + 1

    return start_time, end_time


def convert(seconds: float) -> str:
    min, sec = divmod(seconds, 60)
    return "%02d:%02d" % (min, sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_data_path", default="data/interim/transcripts/")
    parser.add_argument(
        "--answer_data_path",
        default="data/processed/text_predictions/distilbert_squad/",
    )
    parser.add_argument("--converted_data_folder", default="submissions")
    parser.add_argument("--filename", default="test.json")

    args = parser.parse_args()

    with open(f"{args.answer_data_path}/{args.filename}") as fp:
        videos = json.load(fp)

    with open(f"{args.transcript_data_path}/{args.filename}") as fp:
        transcripts = json.load(fp)

    # merge transcript text
    for transcript in transcripts:
        merged = ""
        for line in transcript["transcript"]:
            merged += line["text"] + " "
        transcript["merged"] = merged.strip()

    for video in tqdm(videos):
        transcript_dict = [
            x for x in transcripts if x["video_id"] == video["video_id"]
        ][0]
        start_time, end_time = text_to_time_chars(
            char_position_start=video["prediction"]["start"],
            char_position_end=video["prediction"]["end"],
            transcript=transcript_dict["transcript"],
            merge_type="strict",
        )

        video["prediction"]["start_time"] = convert(start_time)
        video["prediction"]["end_time"] = convert(end_time)

    if not os.path.exists(f"{args.answer_data_path}/{args.converted_data_folder}/"):
        os.makedirs(f"{args.answer_data_path}/{args.converted_data_folder}/")

    with open(
        f"{args.answer_data_path}/{args.converted_data_folder}/{args.filename}", "w"
    ) as fp:
        json.dump(videos, fp, indent=2)
