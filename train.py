from torch import optim
import torch.nn as nn
import torch

import numpy as np
from config import *

#------------prepare enviroment------------
seed = SEED
if seed != None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

torch.cuda.set_device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

from dataset.loading_data import loading_data
from models.ResNet import ResNet50, ResNet101
from torchvision import models
from trainer import Trainer

# Transfer learning from ImageNet
if PRETRAINED:
    net = models.resnet101(pretrained=PRETRAINED)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    #net = models.mobilenet_v3_large(pretrained=True, width_mult=1.0,  reduced_tail=False, dilated=False)
else:
    net = ResNet101(NUM_CLASSES)
net = net.cuda()


optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss().cuda()

#------------Start Training------------
cc_trainer = Trainer(net, loading_data, optimizer, criterion)

start.record()
cc_trainer.forward()
end.record()

torch.cuda.synchronize()
torch.cuda.empty_cache() 

print('Finished Training')
print(start.elapsed_time(end))

