import torch
from torch import optim
import torch.nn as nn
from dataset.loading_data import loading_data
from models.Res50 import ResNet50
import numpy as np
from torchvision import models

from config import *


import random
random.seed(316)


def train(train_loader, val_loader, net):

    min_valid_loss = np.inf
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            # Transform labels to one-hot-encode to scalars
            labels = torch.argmax(labels, dim=2).flatten()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        min_valid_loss = validation(net, epoch, running_loss/len(train_loader), val_loader, min_valid_loss)


def validation(model, epoch, train_loss, val_loader, min_valid_loss):
    valid_loss = 0.0
    model.eval() # Optional when not using Model Specific layer

    for data, labels in val_loader:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        with torch.no_grad():
            # Forward Pass
            predict = model(data)

        # Transform labels to one-hot-encode to scalars
        labels = torch.argmax(labels, dim=2).flatten()
        # Find the Loss
        loss = criterion(predict, labels)
        # Calculate Loss
        valid_loss += loss.item()

    valid_loss = valid_loss / len(val_loader)
    print('Epoch {} - Training loss: {} - Validation Loss: {}'.format(epoch+1, train_loss, valid_loss))

    if min_valid_loss > valid_loss:
        print('Validation Loss Decreased({}--->{})  Saving The Model'.format(min_valid_loss, valid_loss))
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), 'checkpoint.pth')

    return min_valid_loss


# Transfer learning from ImageNet
if PRETRAINED:
    net = models.resnet50(pretrained=PRETRAINED)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, NUM_CLASSES)
else:
    net = ResNet50(NUM_CLASSES)


train_loader, val_loader = loading_data()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

if torch.cuda.is_available():
    net = net.cuda()
    criterion = criterion.cuda()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

train(train_loader, val_loader, net)

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
# 0,221148333 min
