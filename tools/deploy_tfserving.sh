#!/bin/bash

model_path=$1
model_name=$2

sudo docker run -p 8501:8501 -p 8500:8500 -v $model_path:/models/$model_name -e MODEL_NAME=$model_name -t tensorflow/serving

curl localhost:8501/v1/models/$model_name/metadata
