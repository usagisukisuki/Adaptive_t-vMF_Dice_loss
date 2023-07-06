import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image, ImageOps

class KvasirSEG_Dataset(data.Dataset):

    def __init__(self, root=None, dataset_type='train',cross='1', transform=None):
        #self.h_image_size, self.w_image_size = image_size[0], image_size[1]
        self.dataset_type = dataset_type
        self.transform = transform
        self.cross = cross

        self.item_image = np.load(root + "datamodel/{}_data_{}.npy".format(self.dataset_type, self.cross))        
        self.item_gt = np.load(root + "datamodel/{}_label_{}.npy".format(self.dataset_type, self.cross))  
        print(np.bincount(self.item_gt.flatten()))      


    def __getitem__(self, index):
        items_im = self.item_image
        items_gt = self.item_gt
        img_name = items_im[index]
        label_name = items_gt[index]
        label_name = np.where(label_name>200, 1, 0)

        image = Image.fromarray(np.uint8(img_name))
        mask = Image.fromarray(np.uint8(label_name))

        #mask = np.eye(2)[mask]

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self):
        return len(self.item_image)



