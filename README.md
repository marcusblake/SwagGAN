# SwagGAN


## About
This is repository contains the code for my group's final project for Advanced Deep Learning (EECE E6691) taught by Professor Mehmet Turkcan. My group consisted of myself (mab2401), Foo Jit Soon (jf3482), Jiang Guan (jg4329).

The goal of this particular project is to be able to generate fashion images given some input text by a user. The idea is that a user will input an image of some subject and also input a text description of the clothing style that they wish to be worn by the subject, and our model will generate an image of the subject wearing the specified clothing while also keep the subject's pose, facial features, and skin color as intact as possible.

For this project, we closely followed the work done in [FICE](https://arxiv.org/abs/2301.02110) (Pernuš et. al) and our goal was to extend the results to full body images since the original work only focuses up upper body images.

## Setup

To setup your development environment to run our code, we highly recommend using an Anaconda or virtual environment in order to install the necessary packages and Python version >= 3.8. In order to create a new Anaconda environment and install the necessary packages, run the following commands:

```
conda create -n python38 python==3.8
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
```

*Note*: If you are trouble having installing some of the packages with requirements.txt which stops from installing the rest, you can run `cat requirements.txt | xargs -n 1 pip install`.

*Note*: If you ever get the error `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` when running code in this repository, please try to run `pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`.


## Code Structure

### data/
This folder contains the PyTorch data loaders that were necessary for our project. For this project, we worked with the [DeepFashionMultimodal](https://github.com/yumingj/DeepFashion-MultiModal) dataset.

### losses/

This folder contains all of the loss functions that we used for training. In particular, it contains a contrastive loss function that is consistent with what is described in OpenAI's [CLIP](https://arxiv.org/abs/2103.00020). We also implemented separate loss functions that are used when training our entire pipeline together.

### models/
This contains all of the PyTorch neural network modules that are used in our work. It also contains some model configuration files that are necessary for properly loading the pre-trained networks. The code in `fice_encoder` and `fice_stylegan` are directly imported from the FICE project.

### notebooks/
This is where we store all of our Jupyter notebooks which was used to perform exporatory data analysis and verify our work.

### utils/
This contains some useful utilities functions that are used in order to accomplish our work. In particular, most of the work here is related to image stitching, calculating head and body masks given an instance segmentation.

### visualization/
This contains some functions for visualizing work. In particular, it contains the code that we implemented in order to visualize a heatmap of cosine similarities for image and text embeddings.

### train_clip.py
This is the main script for training our clip model. Here's an example of how you an run it:

```
python train_clip.py --lr 1e-4 --dataset_path ~/DeepFashionMultimodal --optim "adam" --checkpoint_frequency 10
```

### train_swaggan.py
This is the main script for training the full pipeline in order to generate an edited image. Here's an example of how you can run it:

```
python train_swaggan.py --lr 0.5 --img_path ./imgs
```

You will need to first download all of the pretrained models to run this script. You can download them by running `./download.sh`. If you want to use a different pre-trained model, then edit the necessary entries in swaggan_config.json.


## Learning Outcomes
This project was quite involved and required us to learn about a few different areas of machine learning. In particular, the areas that this forced us to learn are contrastive representation learning, multimodal learning, and generative adversarial networks. We discovered how difficult it can be to train large scale neural networks.
