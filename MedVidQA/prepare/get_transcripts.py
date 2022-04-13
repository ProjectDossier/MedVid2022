import json
import os

from tqdm.auto import tqdm
from youtube_transcript_api import YouTubeTranscriptApi

in_path = "data/raw/MedVidQA/"
doc_files = [x for x in os.listdir(in_path) if x.endswith("json")]

out_path = "data/interim/transcripts/"
if not os.path.exists(out_path):
    os.makedirs(out_path)


for doc_file in tqdm(doc_files):

    transcripts_list = []

    with open(f"{in_path}/{doc_file}", "r") as rfile:
        data_items = json.load(rfile)

    for item in tqdm(data_items):
        try:
            x = YouTubeTranscriptApi.get_transcript(item["video_id"])
            transcripts_list.append(
                {
                    "video_id": item["video_id"],
                    "sample_id": item["sample_id"],
                    "transcript": x,
                }
            )
        except:
            print(item["sample_id"], item["video_url"])
            transcripts_list.append(
                {
                    "video_id": item["video_id"],
                    "sample_id": item["sample_id"],
                    "transcript": [
                        {"text": "empty transcript", "start": 0.00, "duration": 1.00}
                    ],
                }
            )

    with open(f"{out_path}/{doc_file}", "w") as fp:
        json.dump(transcripts_list, fp, indent=2)
