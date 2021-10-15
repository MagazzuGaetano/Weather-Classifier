import os
from PIL import Image
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from config import *

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
        self.file_name = self.file_name[:800]
        self.num_samples = len(self.file_name)

        self.img_transform = None
        if 'img_transform' in argv.keys():
            self.img_transform = argv['img_transform']
        self.gt_transform = None
        if 'gt_transform' in argv.keys():
            self.gt_transform = argv['gt_transform']

        if self.mode == 'train':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} training images.')
        if self.mode == 'val':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} validation images.')
        

    def __getitem__(self, index):
        img, gt = self.read_image_and_gt(index)

        if self.img_transform != None:
            img = self.img_transform(img)
        if self.gt_transform != None:
            gt = self.gt_transform(gt)

        if self.mode == 'train':
            return img, gt
        elif self.mode == 'val':
            return img, gt
        else:
            print('invalid data mode!!!')

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, index):
        img_path = self.imgfileTemp.format(self.file_name[index].split('.')[0])

        # read image
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        # read ground-truth
        gt = []
        if self.classname == "no weather degradation":
            gt = [1, 0, 0, 0]
        elif self.classname == "fog":
            gt = [0, 1, 0, 0]
        elif self.classname == "rain":
            gt = [0, 0, 1, 0]
        elif self.classname == "snow":
            gt = [0, 0, 0, 1]
        else:
            print('invalid gt!!!')
        gt = np.asarray(gt)

        return img, gt

    def get_num_samples(self):
        return self.num_samples

