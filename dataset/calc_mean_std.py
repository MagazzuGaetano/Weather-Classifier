import os
import cv2
import numpy as np
from math import sqrt


def calc_mean(img_path):
    R_channel = 0
    G_channel = 0
    B_channel = 0
    total_pixel = 0

    images = os.listdir(img_path)
    for i_img, img_name in enumerate(images):
        img = cv2.imread(os.path.join(img_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        total_pixel = total_pixel + img.shape[0] * img.shape[1]
        R_channel = R_channel + np.sum(img[:,:,0])
        G_channel = G_channel + np.sum(img[:,:,1])
        B_channel = B_channel + np.sum(img[:,:,2])
    R_mean = R_channel / total_pixel
    G_mean = G_channel / total_pixel
    B_mean = B_channel / total_pixel
    return [R_mean, G_mean, B_mean], total_pixel

def calc_std(img_path, mean, total_pixel):
    images = os.listdir(img_path)
    R_mean, G_mean, B_mean = mean

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for i_img, img_name in enumerate(images):
        img = cv2.imread(os.path.join(img_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

    R_std = sqrt(R_channel / total_pixel)
    G_std = sqrt(G_channel / total_pixel)
    B_std = sqrt(B_channel / total_pixel)
    return [R_std, G_std, B_std]

def calc_mean_std():
    classnames = ["no weather degradation", "fog", "rain", "snow"]

    for classname in classnames:
        img_path = './preprocessed_data/train/{}'.format(classname)

        mean, total_pixel = calc_mean(img_path)
        std = calc_std(img_path, mean, total_pixel)

        R_mean, G_mean, B_mean = mean
        R_std, G_std, B_std = std
        print("mean: [{}, {}, {}]".format(R_mean/255, G_mean/255, B_mean/255))
        print("stdev: [{}, {}, {}]".format(R_std/255, G_std/255, B_std/255))

calc_mean_std()
