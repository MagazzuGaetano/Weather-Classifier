import os
import numpy as np
import cv2

# This code is to create the dataset probably i don't need anymore in this project!
def preprocess_data():
    mode = 'val'
    img_path = '../data_processed/{}/img'.format(mode)
    out_path = './dataset'
    images = os.listdir(img_path)

    with open('../data_processed/{}/image_labels.txt'.format(mode)) as f:
        lines = f.readlines()

    for info in lines:
        info = info.split(',')
        img_name = info[0]
        weather_code = info[3]

        img = cv2.imread(os.path.join(img_path, img_name + '.jpg'))
        print(os.path.join(img_path, img_name + '.jpg'))

        if weather_code == "0":
            weather_conditions = 'no weather degradation'
        elif weather_code == "1":
            weather_conditions = 'fog'
        elif weather_code == "2":
            weather_conditions = 'rain'
        else:
            weather_conditions = 'snow'

        cv2.imwrite(os.path.join(out_path, weather_conditions, img_name + '.jpg'), img)

