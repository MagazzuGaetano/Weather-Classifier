import torch
from torch import optim
import torch.nn as nn
from dataset.loading_data import loading_data
from models.ResNet import ResNet50
import numpy as np
from torchvision import models
from config import *

train_loader, val_loader = loading_data()


# Transfer learning from ImageNet
if PRETRAINED:
    net = models.resnet50(pretrained=PRETRAINED)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, NUM_CLASSES)
else:
    net = ResNet50(NUM_CLASSES)
net = net.cuda()


criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('Epoch {} - Training loss: {}'.format(epoch+1, running_loss/len(train_loader)))

# whatever you are timing goes here
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print('Finished Training')
print(start.elapsed_time(end))  # milliseconds


# save
PATH = './classifier.pth'
torch.save(net.state_dict(), PATH)

# 11034015.0ms ~ 3 ore