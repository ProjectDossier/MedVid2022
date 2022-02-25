#!/bin/bash



DATASETS="test val train"
MODELS="bert specter multiqa_minilm msmarco_roberta bluebert nli_mpnet"
for dataset in ${DATASETS}
do
  for model in ${MODELS}
  do
    echo ${dataset} ${model}
    python MedVidQA/model/passage_similarity.py --model ${model} --dataset ${dataset} --transcript_data_path data/interim/ocr/
  done
done


