from torch.util.data import Dataset
from google.cloud import storage



class CloudStorageDataset:
    def __init__(self, cloud_bucket):
        self.cloud_bucket = cloud_bucket

    def get_from_cloud_storage(self, filename):
        client = storage.Client()
        blob = client.



class DeepFashionMultimodalImageAndTextDataset(Dataset, CloudStorageDataset):
    def __init__(self, dataset_folder, transforms = None, women_only = False, men_only = False, cloud_bucket = None):
        self.folder = dataset_folder


    def __getitem__(self, idx):
        pass


    def __len__(self):
        pass
        


class DeepFashionImageSegementationDataset(Dataset, CloudStorageDataset):
    def __init__(self, dataset_folder, transforms = None, women_only = False, men_only = False, cloud_bucket = None):
        self.folder = dataset_folder
        self.segmentation_folder = f'{dataset_folder}/segm'
        self.images

    def __getitem__(self, idx):
        pass


    def __len__(self):
        pass