# test
import torch
from models.ResNet import ResNet50 as net
from config import *
import os
import numpy as np
import random
import torchvision.transforms as standard_transforms
from PIL import Image
from torch.autograd import Variable
from sklearn.metrics import f1_score
from config import *


seed = SEED
if seed != None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

torch.cuda.set_device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*MEAN_STD)
])

gt_transform = standard_transforms.Compose([
    standard_transforms.ToTensor()
])

# reload
PATH = './latest_state.pth'
net = net(NUM_CLASSES)
net.load_state_dict(torch.load(PATH), strict=False)
net.eval()

correct = 0
labels = []
predicted = []

for classname in CLASSES:
    input_path = "./preprocessed_data/test/{}".format(classname)
    images = os.listdir(input_path)
    random.shuffle(images)

    for i_img, img_name in enumerate(images):
        # read image
        img = Image.open(os.path.join(input_path, img_name))
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img.resize(TRAIN_SIZE)
        img = img_transform(img)

        # read ground-truth
        gt = []
        if classname == "no weather degradation":
            gt = [1, 0, 0, 0]
        elif classname == "fog":
            gt = [0, 1, 0, 0]
        elif classname == "rain":
            gt = [0, 0, 1, 0]
        elif classname == "snow":
            gt = [0, 0, 0, 1]
        else:
            print('invalid gt!!!')
        gt = np.asarray(gt).reshape((1, NUM_CLASSES))
        gt = gt_transform(gt)

        # Disable grad
        with torch.no_grad():
            img = Variable(img[None,:,:,:])            
            output = net(img)

            if np.argmax(gt) == np.argmax(output):
                correct = correct + 1
            else:
                print('gt: {}, pred: {}'.format(np.argmax(gt), np.argmax(output)))

            labels.append(np.argmax(gt))
            predicted.append(np.argmax(output))


total_images = 1533
print("corrected classified: {} / {}".format(correct, total_images))
print("F1: {}".format(f1_score(labels, predicted, average='micro')))
print("F1 (unbalanced): {}".format(f1_score(labels, predicted, average='weighted')))
print("F1 (per class): {}".format(f1_score(labels, predicted, average=None)))





