# test
import torch
from models.ResNet import ResNet50, ResNet101
from config import *
import os
import numpy as np
import random
import torchvision.transforms as standard_transforms
from PIL import Image
from torch.autograd import Variable
from config import *
from dataset.label_preprocess import *
from sklearn.metrics import f1_score

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
PATH = 'checkpoint.pth' #'./trained_models/res50_F1_0.8395_epochs_150_batch_8.pth'
net = ResNet101(NUM_CLASSES).cuda()
model = torch.load(PATH)
if 'net' in model.keys():
    net.load_state_dict(model['net'], strict=False)
else:
    net.load_state_dict(model, strict=False)
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

            labels.append(label)
            predicted.append(output)

total_images = 1533
print("corrected classified: {} / {}".format(correct, total_images))
print("F1: {}".format(f1_score(labels, predicted, average='micro')))
print("F1 (per class): {}".format(f1_score(labels, predicted, average=None)))
#print("F1 (unbalanced): {}".format(f1_score(labels, predicted, average='weighted')))
