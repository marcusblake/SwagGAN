from torch.utils.data import Dataset
from PIL import Image
import json
import re


def get_images_and_text_matching_regex(captions, prefix):
    images, descriptions = [], []
    for image_file, caption in captions.items():
        if image_file.startswith(prefix):
            images.append(image_file)
            descriptions.append(caption)
    return images, descriptions

class DeepFashionMultimodalImageAndTextDataset(Dataset):
    def __init__(self, dataset_folder,
                 text_transforms = None,
                 image_transforms = None,
                 women_only = False,
                 men_only = False):
        self.folder = dataset_folder
        self.images_folder = f'{dataset_folder}/images'
        
        self.images = []
        self.captions = []
        with open(f'{dataset_folder}/captions.json') as f:
            captions = json.loads(f.read())
            if men_only:
                self.images, self.captions = get_images_and_text_matching_regex(captions, 'MEN')
            elif women_only:
                self.images, self.captions = get_images_and_text_matching_regex(captions, 'WOMEN')
            else:
                self.images, self.captions = list(captions.keys()), list(captions.values())

    def __getitem__(self, idx):
        result = {}
        result['img'] = Image.open(f'{self.images_folder}/{self.images[idx]}').convert('RGB').resize((256,256))
        result['txt'] = self.captions[idx]
        return result

    def __len__(self):
        return len(self.images)

class DeepFashionImageSegementationDataset(Dataset):
    def __init__(self, dataset_folder, transforms = None, women_only = False, men_only = False, cloud_bucket = None):
        self.folder = dataset_folder
        self.cloud_bucket = cloud_bucket
        self.segmentation_folder = f'{dataset_folder}/segm'
        self.images_folder = f'{dataset_folder}/images'

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

class DeepFashionImagePoseDataset(Dataset):
    def __init__(self, dataset_folder, transforms = None, women_only = False, men_only = False, cloud_bucket = None):
        self.folder = dataset_folder
        self.cloud_bucket = cloud_bucket
        self.denspose_folder = f'{dataset_folder}/densepose'
        self.images_folder = f'{dataset_folder}/images'

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass