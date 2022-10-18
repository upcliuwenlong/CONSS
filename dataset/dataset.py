import random
from torch.utils.data import Dataset
import warnings
import numpy as np
import os
import math
from albumentations import *
warnings.filterwarnings('ignore')

class UnlabelDataset(Dataset):
    def __init__(self,data, labels, augmentations,slice_width=256,sampling_pos=None):
        self.data = data
        self.labels = labels
        self.sampling_pos = sampling_pos
        self.slice_width = slice_width
        self.augmentations = augmentations
        self.images = list()
        self.masks = list()
        self.make_dataset()

    def add_data(self,image,mask):
        image = np.expand_dims(image,-1)
        auged_data = self.augmentations(
            image=image,
            mask=mask
        )
        auged_image = auged_data['image']
        auged_mask = auged_data['mask']
        auged_image = auged_image.transpose(2, 0, 1)
        self.images.append(auged_image)
        self.masks.append(auged_mask)
    def make_dataset(self):
        if self.sampling_pos is not None:
            all_sample_pos = range(self.data.shape[2])
            unlabel_sample_pos = set(all_sample_pos) - set(self.sampling_pos)
        else:
            unlabel_sample_pos = range(self.data.shape[2])

        # inline
        for step in range(int(1 + (self.data.shape[1]-self.slice_width)/(self.slice_width))):
            x_start = int(step * (self.slice_width))
            for y in unlabel_sample_pos:
                mask = self.labels[:, x_start:x_start + self.slice_width, y].astype(np.int64)
                image = self.data[:, x_start:x_start+self.slice_width, y]
                self.add_data(image, mask)

        if self.data.shape[1]-self.slice_width > 0:
            for y in unlabel_sample_pos:
                mask = self.labels[:, self.data.shape[1]-self.slice_width:self.data.shape[1], y].astype(np.int64)
                image = self.data[:, self.data.shape[1]-self.slice_width:self.data.shape[1], y]
                self.add_data(image,mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class FewLabelDataset(Dataset):
    def __init__(self,data, labels, augmentations,slice_width=256,sampling_pos=None,sparse=False):
        self.data = data
        self.labels = labels
        self.sampling_pos = sampling_pos
        self.sparse = sparse
        self.slice_width = slice_width
        self.augmentations = augmentations
        self.images = list()
        self.masks = list()
        self.make_dataset()

    def over_sample(self,unlabel_num):
        zip_data = list(zip(self.images,self.masks))
        repeats = math.ceil(unlabel_num / len(self.images))
        zip_data = zip_data*repeats
        over_smp_zip = random.sample(zip_data,unlabel_num)
        images,masks = zip(*over_smp_zip)
        self.images = images
        self.masks = masks

    def add_data(self, image, mask):
        image = np.expand_dims(image, -1)
        auged_data = self.augmentations(
            image=image,
            mask=mask
        )
        auged_image = auged_data['image']
        auged_mask = auged_data['mask']
        if self.sparse:
            h,w  = auged_mask.shape
            ct = CoarseDropout(max_holes=10,min_holes=10,max_height=h,max_width=16,fill_value=-1,p=1)
            auged_mask = ct(image=auged_mask)['image']
        auged_image = auged_image.transpose(2, 0, 1)
        self.images.append(auged_image)
        self.masks.append(auged_mask)
    def make_dataset(self):
        label_sample_pos = self.sampling_pos
        print('******label_sample_position*****',label_sample_pos)

        # inline
        for step in range(int(1 + (self.data.shape[1]-self.slice_width)/(self.slice_width))):
            x_start = int(step * (self.slice_width))
            for y in label_sample_pos:
                mask = self.labels[:, x_start:x_start + self.slice_width, y].astype(np.int64)
                image = self.data[:, x_start:x_start+self.slice_width, y]
                self.add_data(image, mask)

        if self.data.shape[1]-self.slice_width > 0:
            for y in label_sample_pos:
                mask = self.labels[:, self.data.shape[1]-self.slice_width:self.data.shape[1], y].astype(np.int64)
                image = self.data[:, self.data.shape[1]-self.slice_width:self.data.shape[1], y]
                self.add_data(image,mask)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

