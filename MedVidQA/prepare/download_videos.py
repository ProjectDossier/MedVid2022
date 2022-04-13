import json
import os

import pytube.exceptions
from pytube import YouTube
from tqdm.notebook import tqdm

in_path = "../data/raw/MedVidQA/"
doc_files = [x for x in os.listdir(in_path) if x.endswith("json")]

out_path = "../data/interim/videos/"
if not os.path.exists(out_path):
    os.makedirs(out_path)

for doc_file in tqdm(doc_files):
    with open(f"{in_path}/{doc_file}", "r") as rfile:
        data_items = json.load(rfile)

    print(len(data_items))

    for item in tqdm(data_items):
        # print(item['video_link'])
        try:
            YouTube(item["video_url"]).streams.filter(
                file_extension="mp4"
            ).first().download(output_path=out_path, filename=f"{item['video_id']}.mp4")
        except pytube.exceptions.VideoPrivate:
            print(item["video_url"])
