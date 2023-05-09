#!/bin/bash

MODEL_DIRECTORY="models/pre_trained_models"
declare -a PretrainedModelUrls=(
    "http://storage.googleapis.com/deep_fashion_multimodal/pre_trained_models/checkpoint_0040_DeepLabV3_Fashion_Men.pth"
    "http://storage.googleapis.com/deep_fashion_multimodal/pre_trained_models/checkpoint256_0040_DeepLabV3_Fashion_Men.pth"
    "http://storage.googleapis.com/deep_fashion_multimodal/pre_trained_models/e4e.pt"
    "http://storage.googleapis.com/deep_fashion_multimodal/pre_trained_models/generator.pt"
    "http://storage.googleapis.com/deep_fashion_multimodal/pre_trained_models/e4e_stylegan256x256.pt"
    "http://storage.googleapis.com/deep_fashion_multimodal/pre_trained_models/stylegan-deepfashion256x256.pt"
    "http://storage.googleapis.com/deep_fashion_multimodal/pre_trained_models/clip_model_epoch5.pt"
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


