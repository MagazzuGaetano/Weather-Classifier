import os
from PIL import Image
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from config import *
from dataset.label_preprocess import get_label

class WeatherDataset(Dataset):
    def __init__(self, data_path, datasetname, mode, classname, **argv):
        self.mode = mode
        self.data_path = data_path
        self.datasetname = datasetname
        self.classname = classname

        self.image_path = os.path.join(data_path, mode, classname)
        self.imgfileTemp = os.path.join(self.image_path, '{}.jpg')

        self.file_name = [filename for filename in os.listdir(self.image_path) \
                if os.path.isfile(os.path.join(self.image_path, filename))]
        self.file_name = self.file_name[:100] #850
        self.num_samples = len(self.file_name)

        self.img_transform = None
        if 'img_transform' in argv.keys():
            self.img_transform = argv['img_transform']
        self.label_transform = None
        if 'label_transform' in argv.keys():
            self.label_transform = argv['label_transform']

        if self.mode == 'train':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} training images.')
        if self.mode == 'val':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} validation images.')


    def __getitem__(self, index):
        img, label = self.read_image_and_label(index)

        if self.img_transform != None:
            img = self.img_transform(img)
        if self.label_transform != None:
            label = self.label_transform(label)

        if self.mode == 'train':
            return img, label
        elif self.mode == 'val':
            return img, label
        else:
            print('invalid data mode!!!')

    def __len__(self):
        return self.num_samples

    def read_image_and_label(self, index):
        img_path = self.imgfileTemp.format(self.file_name[index].split('.')[0])

        # read image
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        # read ground-truth
        label = get_label(self.classname)

        return img, label

    def get_num_samples(self):
        return self.num_samples

