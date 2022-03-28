import argparse
import json
import os

import pytesseract
from PIL import Image
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="data/interim/images/")
    parser.add_argument("--ocr_path", default="data/interim/ocr/")

    args = parser.parse_args()

    image_folders = [
        x for x in os.listdir(args.in_path) if os.path.isdir(f"{args.in_path}/{x}")
    ]

    if not os.path.exists(args.ocr_path):
        os.makedirs(args.ocr_path)

    out_list = []
    for folder in tqdm(image_folders):
        transcript = []
        for image in os.listdir(f"{args.in_path}/{folder}"):
            ocred_text = pytesseract.image_to_string(
                Image.open(f"{args.in_path}/{folder}/{image}")
            )
            frame_n = image.split(".")[0].split("_")[1]

            ocred_text = " ".join(ocred_text.split())

            if ocred_text:
                transcript.append(
                    {"text": ocred_text, "start": float(frame_n), "duration": 3.0}
                )

        transcript_sorted = sorted(transcript, key=lambda d: d["start"])

        out_list.append({"video_id": folder, "transcript": transcript_sorted})

        with open(
            f"{args.ocr_path}/ocr.json",
            "w",
        ) as fp:
            json.dump(out_list, fp, indent=2)
