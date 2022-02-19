import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image, ImageOps

class CVCClinicDB_Dataset(data.Dataset):
    def __init__(self, dataset_type='train', transform=None):
        self.item_image = sorted(os.listdir("Dataset/archive/PNG/Original"))
        self.item_gt = sorted(os.listdir("Dataset/archive/PNG/GroundTruth"))

        self.transform = transform

        if dataset_type=='train':
            self.images = self.item_image[:368]
            self.labels = self.item_gt[:368]
        elif dataset_type=='val':
            self.images = self.item_image[368:490]
            self.labels = self.item_gt[368:490]
        elif dataset_type=='test':
            self.images = self.item_image[490:]
            self.labels = self.item_gt[490:]

    def __getitem__(self, index):
        img_name = self.images[index]
        label_name = self.labels[index]
        image = Image.open("Dataset/archive/PNG/Original/" + img_name).convert("RGB")
        label = Image.open("Dataset/archive/PNG/GroundTruth/" + label_name).convert("L")
        label = np.array(label)
        mask = np.where(label>200, 1, 0)
        mask = Image.fromarray(np.uint8(mask))

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self):
        return len(self.images)


