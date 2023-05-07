#!/bin/bash

MODEL_DIRECTORY="models/pre_trained_models"
declare -a PretrainedModelUrls=(
    "http://storage.googleapis.com/deep_fashion_multimodal/pre_trained_models/checkpoint_0040_DeepLabV3_Fashion_Men.pth"
)

if [ -d ${MODEL_DIRECTORY} ]; then 
    echo "Pre trained models directory exists. Attemping to download pre-trained models."
else
    echo "Pre trained models directory doesn't exist. Creating..."
    mkdir ${MODEL_DIRECTORY}
    echo "Directory models/pre_trained_models created"
fi

cd ${MODEL_DIRECTORY}

for modelUrl in "${PretrainedModelUrls[@]}"
do 
    wget ${modelUrl}
done


