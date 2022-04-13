# A Dataset for Medical Instructional Video Classification and Question Answering

These are the benchmark experiments reported for MedVidQA dataset in our paper [A Dataset for Medical Instructional Video Classification and Question Answering](https://arxiv.org/pdf/2201.12888.pdf)


Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follow:
```shell script
# preparing environment
conda env create -f environment.yml
conda activate medvidqa
```

## Data Preparation
1) Download the MedVidQA dataset from [OSF repository](https://doi.org/10.17605/OSF.IO/PC594) and place train.json/val.json/test.json in `data/dataset/medvidqa` directory
2) Download the video features from [here](https://bionlp.nlm.nih.gov/VideoFeatures.zip), unzip the file and place the content of `MedVidQA/I3D` in `data/features/medvidqa`
3) Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it to `data/word_embedding`

If you want to prepare your own video features, please follow these steps:
1) Download the pre-trained RGB model from [here](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt) and place it in `data` directory
2) set the pythonpath
```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/MedVidQA/directory
```
3) Run the following command

``python prepare/extract_medvidqa.py
``


## Training and Test

```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medvidqa/directory
python main.py --mode train
python main.py --mode test
```

## Credit
This code repo is adapted from this [repo](https://github.com/IsaacChanghau/VSLNet).


## Replication steps


1. Download videos: `prepare/download_videos.py` 
2. Download transcripts: `prepare/get_transcripts.py`
3. Get screenshots every 3 seconds: `prepare/get_video_frames.py` 
4. Find text in the video frames: `prepare/ocr_images.py`
5. Merge transcripts to create more data points: `prepare/expand_data.py`
6. Calculate the question-passage similarity: `model/passage_similarity.py`  
7. Normalise similarity scores for all input features for each model `util/normalise_scores.py`
8. Merge relevance scores from different models: `util/merge_relevance.py`
9. Create plots for visual check: `visualisations/plot_relevance.py`  
10. Run Models:
    - Extractive Q&A baseline: `model/extractive_qa.py`
    - Multi-output regression:
      - perform time normalisation: `util/time_normalisation.py` 
      - `model/multi_output_regression.py`
    - Peak detection: `model/peak_detection.py`
