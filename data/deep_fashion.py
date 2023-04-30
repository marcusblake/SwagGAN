from torch.utils.data import Dataset
from PIL import Image
from .cloud_utils import get_from_cloud_storage

class DeepFashionMultimodalImageAndTextDataset(Dataset):
    def __init__(self, dataset_folder, transforms = None, women_only = False, men_only = False, cloud_bucket = None):
        self.folder = dataset_folder
        self.cloud_bucket = cloud_bucket

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

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