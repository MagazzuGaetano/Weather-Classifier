import os
import torchvision.transforms as standard_transforms
from torchvision.transforms.transforms import RandomHorizontalFlip
from .dataset import WeatherDataset
from torch.utils.data import DataLoader, ConcatDataset
from config import *


def createTrainData(datasetname, Dataset):
    img_transform = standard_transforms.Compose([
        standard_transforms.RandomHorizontalFlip(),
    	standard_transforms.RandomCrop(TRAIN_SIZE),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*MEAN_STD)
    ])

    gt_transform = None

    train_set_0 = Dataset(DATA_PATH,datasetname,'train','no weather degradation',img_transform=img_transform,gt_transform=gt_transform)
    train_set_1 = Dataset(DATA_PATH,datasetname,'train','fog',img_transform=img_transform,gt_transform = gt_transform)
    train_set_2 = Dataset(DATA_PATH,datasetname,'train','rain',img_transform=img_transform,gt_transform=gt_transform)
    train_set_3 = Dataset(DATA_PATH,datasetname,'train','snow',img_transform=img_transform,gt_transform=gt_transform)
    train_set = ConcatDataset([train_set_0, train_set_1, train_set_2, train_set_3])
    return DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, num_workers=2, shuffle=True, drop_last=True)

def createValData(datasetname, Dataset):
    img_transform = standard_transforms.Compose([
    	standard_transforms.RandomCrop(TRAIN_SIZE),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*MEAN_STD)
    ])

    gt_transform = None

    val_set_0 = Dataset(DATA_PATH,datasetname,'val','no weather degradation',img_transform=img_transform,gt_transform=gt_transform)
    val_set_1 = Dataset(DATA_PATH,datasetname,'val','fog',img_transform=img_transform,gt_transform=gt_transform)
    val_set_2 = Dataset(DATA_PATH,datasetname,'val','rain',img_transform=img_transform,gt_transform=gt_transform)
    val_set_3 = Dataset(DATA_PATH,datasetname,'val','snow',img_transform=img_transform,gt_transform=gt_transform)
    val_set = ConcatDataset([val_set_0, val_set_1, val_set_2, val_set_3])
    return DataLoader(val_set, batch_size=VAL_BATCH_SIZE, num_workers=2, shuffle=True, drop_last=True)

def loading_data():
    train_loader = createTrainData('WeatherDataset', WeatherDataset)
    val_loader = createValData('WeatherDataset', WeatherDataset)
    return train_loader, val_loader

