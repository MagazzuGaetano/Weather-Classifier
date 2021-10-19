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
from config import *
from dataset.label_preprocess import *

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


# reload
PATH = 'classifier.pth' #'./trained_models/res50_F1_0.8395.pth'
net = net(NUM_CLASSES).cuda()
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
        label = get_label(classname)

        # Disable grad
        with torch.no_grad():
            img = img[None,:,:,:].cuda()

            output = net(img).cpu()
            output = torch.nn.functional.softmax(output[0], dim=0)
            output = torch.argmax(output, dim=0)
            #output = torch.max(k.data, 1)[1].item() # cosi non sono probabilità!

            if label == output:
                correct = correct + 1
            else:
                print('label: {}, pred: {}'.format(label, output))

            labels.append(np.argmax(label))
            predicted.append(np.argmax(output))


total_images = 1533
print("corrected classified: {} / {}".format(correct, total_images))
print("F1: {}".format(correct/total_images))





