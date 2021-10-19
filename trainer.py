import torch
from torch import optim
import torch.nn as nn
import numpy as np
from config import *


class Trainer():
    def __init__(self, net, dataloader, optimizer, criterion):
        
        self.net = net
        self.train_loader, self.val_loader = dataloader()
        self.optimizer = optimizer      
        self.criterion = criterion

        self.epoch = 0
        self.i_tb = 0
        self.min_valid_loss = np.inf

        if RESUME:
            latest_state = torch.load(RESUME_PATH)
            self.net.load_state_dict(latest_state['net'], strict=False)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.min_valid_loss = latest_state['min_valid_loss']


    def forward(self):
        for epoch in range(self.epoch, MAX_EPOCH):  # loop over the dataset multiple times
            self.epoch = epoch

            train_loss = self.train()
            self.validate(train_loss)

        self.save_model('latest_state.pth') # save final model


    def train(self):
        self.net.train()
        running_loss = 0.0

        for i, data in enumerate(self.train_loader, 0):
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

            self.i_tb += 1

        #print('Epoch {} - Training loss: {}'.format(self.epoch+1, running_loss/len(self.train_loader)))
        return running_loss/len(self.train_loader)


    def validate(self, train_loss):
        self.net.eval() # Optional when not using Model Specific layer
        valid_loss = 0.0

        for data, labels in self.val_loader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            with torch.no_grad():
                # Forward Pass
                predict = self.net(data)

            # Find the Loss
            loss = self.criterion(predict, labels)
            # Calculate Loss
            valid_loss += loss.item()

        valid_loss = valid_loss / len(self.val_loader)
        print('Epoch {} - Training loss: {} - Validation Loss: {}'.format(self.epoch+1, train_loss, valid_loss))
    
        self.save_checkpoint(valid_loss)


    def save_checkpoint(self, valid_loss):

        if self.min_valid_loss > valid_loss:
            print('Validation Loss Decreased({}--->{})  Saving The Model'.format(self.min_valid_loss, valid_loss))

            self.min_valid_loss = valid_loss
            self.save_model('checkpoint.pth') # save checkpoint


    def save_model(self, model_name):
        latest_state = {
            'net': self.net.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'epoch': self.epoch,
            'i_tb': self.i_tb,
            'min_valid_loss': self.min_valid_loss,
            'mean_std': MEAN_STD
        }

        # Saving State Dict
        torch.save(latest_state, model_name)