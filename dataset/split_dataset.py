import os
import cv2
import numpy as np
import random

classnames = ["no weather degradation", "fog", "rain", "snow"]
modes = ["train", "val", "test"]

for classname in classnames:
    input_path = "./jhucrowd+weather dataset/{}".format(classname)
    images = os.listdir(input_path)
    random.shuffle(images)

    N = len(images)
    tot_train = int(N * 0.7)
    tot_val = int(N * 0.1)
    tot_test = int(N * 0.2)

    r = N - (tot_train + tot_val + tot_test)
    tot_train = tot_train + r

    start_index_train = 0
    start_index_val = tot_train
    start_index_test = tot_train + tot_val

    for i_img, img_name in enumerate(images):

        if i_img < start_index_val:
            mode = modes[0]
        elif i_img < start_index_test and i_img >= start_index_val:
            mode = modes[1]
        else:
            mode = modes[2]

        output_path = "./preprocessed_data/{}/{}".format(mode, classname)
        print(os.path.join(output_path, img_name))

        image = cv2.imread(os.path.join(input_path, img_name))
        cv2.imwrite(os.path.join(output_path, img_name), image)
