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
from models.ResNet import ResNet50 as Model
from torchvision import models
from trainer import Trainer


# Transfer learning from ImageNet
if PRETRAINED:
    net = models.resnet50(pretrained=PRETRAINED)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, NUM_CLASSES)
else:
    net = Model(NUM_CLASSES)
net = net.cuda()

optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss().cuda()

#------------Start Training------------
cc_trainer = Trainer(net, loading_data, optimizer, criterion)

start.record()
cc_trainer.forward()
end.record()

torch.cuda.synchronize()
print('Finished Training')
print(start.elapsed_time(end))

PATH = './latest_state.pth'
latest_state = {
    'net': cc_trainer.net.state_dict(),
    'optimizer':cc_trainer.optimizer.state_dict(),
    'epoch': cc_trainer.epoch,
    'i_tb': cc_trainer.i_tb,
    'min_valid_loss': cc_trainer.min_valid_loss,
    'mean_std': MEAN_STD
}
torch.save(latest_state, PATH)
