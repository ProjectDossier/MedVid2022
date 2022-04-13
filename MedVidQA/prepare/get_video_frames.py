import os

import cv2
from tqdm.notebook import tqdm

in_path = "../data/interim/videos/"
out_path = "../data/interim/images/"

if not os.path.exists(out_path):
    os.makedirs(out_path)


video_files = [x for x in os.listdir(in_path) if x.endswith("mp4")]
len(video_files)

every_n_seconds = 3
for video_file in tqdm(video_files):
    in_file = f"{in_path}/{video_file}"

    out_folder = f"{out_path}/{video_file.split('.')[0]}"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    vidcap = cv2.VideoCapture(in_file)
    count = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    while success:
        success, image = vidcap.read()
        if (
            count % (every_n_seconds * fps) == fps
        ):  # every n seconds + 1 as we don't care about second 0 but actually second 1
            try:
                cv2.imwrite(f"{out_folder}/frame_{count/fps}.jpg", image)
            except:
                print("fail to save")
        count += 1
