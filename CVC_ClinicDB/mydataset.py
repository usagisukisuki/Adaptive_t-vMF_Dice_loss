import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image, ImageOps

class CVCClinicDB_Dataset(data.Dataset):

    def __init__(self, root=None, dataset_type='train',cross='1', transform=None):
        self.dataset_type = dataset_type
        self.transform = transform
        self.cross = cross

        self.item_image = np.load(root + "datamodel/{}_data_{}.npy".format(self.dataset_type, self.cross))        
        self.item_gt = np.load(root + "datamodel/{}_label_{}.npy".format(self.dataset_type, self.cross))  


    def __getitem__(self, index):
        img_name = self.item_image[index]
        label_name = self.item_gt[index]
        label_name = np.where(label_name>200, 1, 0)

        image = Image.fromarray(np.uint8(img_name))
        label = Image.fromarray(np.uint8(label_name))

        if self.transform:
            image,label = self.transform(image, label)

        return image,label

    def __len__(self):
        return len(self.item_image)


