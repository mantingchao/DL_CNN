#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:23:55 2021

@author: Manting
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import cv2

#%% read data
train_data = pd.read_csv("train.csv") 
test_data = pd.read_csv("test.csv")

#%% data
def data_init(data):
    data_list =[]
    transform1 = transforms.Compose([
        transforms.ToTensor(), 
        ])
    
    for i in range(len(data)):
        name = data.iloc[i].file_name
        img = cv2.imread('images/' + name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c_img = cv2.resize(img,dsize = (32, 32), interpolation = cv2.INTER_AREA)
        tensor = transform1(c_img)
        data_list.append(tensor)
    print(data_list)
    return data_list

#%% cropping data
def crop_data(data):
    data_list =[]
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        ])
    
    for i in range(len(data)):
        name = data.iloc[i].file_name
        img = cv2.imread('images/' + name)
        img = img[data.iloc[i].y1: data.iloc[i].y2, data.iloc[i].x1: data.iloc[i].x2]
        plt.imshow(img)
    
        c_img = cv2.resize(img,dsize = (32, 32), interpolation = cv2.INTER_AREA)
        tensor = transform1(c_img)
        data_list.append(tensor)
    return data_list

#%% CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding = 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(8 * 8 * 64 , 64)
        self.fc2 = nn.Linear(64, 2)
        self.fc3 = nn.Linear(2, 7)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.view(-1, 8 * 8 * 64)
        x = self.fc1(x)
        x = F.relu(x)
        self.output = self.fc2(x)
        x = self.fc3(self.output)
        
        return x

#%%
def plot(train_loss, train_acc, test_loss, test_acc):
    plt.title("learning curves")
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    
    plt.title("accuracy rate")
    plt.plot(train_acc, label='train accuracy')
    plt.plot(test_acc, label='test accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    
#%%
def scatter_plot(data, y_test):
    print(data,y_test)
    data = pd.DataFrame(data.numpy(), columns = ['x', 'y'])
    y_test = pd.DataFrame(y_test.numpy())
    data['label'] = y_test
    data.sort_values(by=['label'], inplace = True)
    print(data)
    colors = {0:'r', 1:'b', 2:'g', 3:'c', 4:'m', 5:'y', 6:'k'}
    
    fig, ax = plt.subplots()
    grouped = data.groupby('label')
    for key, group in grouped:
        group.plot(ax = ax, kind='scatter', x = 'x', y = 'y', label = key, color = colors[key])
    plt.show()

#%% data
batch_size = 16

x_train = torch.stack(data_init(train_data))
x_test =  torch.stack(data_init(test_data))
y_train = torch.from_numpy(np.asarray(train_data['category']))
y_test = torch.from_numpy(np.asarray(test_data['category'])) 

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(train, batch_size)
test = TensorDataset(x_test, y_test)
test_loader = DataLoader(test, len(test_data))

#%%
net = CNN()
#print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)


train_loss = []
train_acc = []
test_loss = []
test_acc = []
scatter_epoch = []
for epoch in range(100):
    train_correct = 0
    train_total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, train_labels = data
        optimizer.zero_grad()
        outputs = net(inputs) # (n_samples, channels, height, width)
        # train loss
        loss = criterion(outputs, train_labels)
        
        # accuracy
        _, train_predicted = torch.max(outputs.data, 1)
        train_total += train_labels.size(0)
        train_correct += (train_predicted == train_labels).sum().item()
        # backward
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())  
    train_acc.append(100 * train_correct/ train_total)
    print('Train Accuracy:',100 * train_correct/ train_total)
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            images, test_labels = data
            outputs = net(images)
            # test loss
            loss = criterion(outputs, test_labels)
            # accuracy
            _, test_predicted = torch.max(outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()
    if epoch == 20:
        scatter_epoch.append([net.output, y_test])
    elif epoch == 70:
        scatter_epoch.append([net.output, y_test])
    test_loss.append(loss.item())
    test_acc.append(100 * test_correct/ test_total)
    print('Test Accuracy:',100 * test_correct/ test_total)

plot(train_loss, train_acc, test_loss, test_acc) 

for i in scatter_epoch:
    scatter_plot(i[0], i[1])
    

#%% 2. crapping
batch_size = 16

x_train = torch.stack(crop_data(train_data))
x_test =  torch.stack(crop_data(test_data))
y_train = torch.from_numpy(np.asarray(train_data['category']))
y_test = torch.from_numpy(np.asarray(test_data['category'])) 

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(train, batch_size)
test = TensorDataset(x_test, y_test)
test_loader = DataLoader(test, len(test_data))

#%%
net = CNN()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)


train_loss = []
train_acc = []
test_loss = []
test_acc = []
scatter_epoch = []
for epoch in range(100):
    train_correct = 0
    train_total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, train_labels = data
        optimizer.zero_grad()
        outputs = net(inputs) # (n_samples, channels, height, width)
        # train loss
        loss = criterion(outputs, train_labels)
        
        # accuracy
        _, train_predicted = torch.max(outputs.data, 1)
        train_total += train_labels.size(0)
        train_correct += (train_predicted == train_labels).sum().item()
        # backward
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())  
    train_acc.append(100 * train_correct/ train_total)
    #print('Train Accuracy:',100 * train_correct/ train_total)
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            images, test_labels = data
            outputs = net(images)
            # test loss
            loss = criterion(outputs, test_labels)
            # accuracy
            _, test_predicted = torch.max(outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()
    if epoch == 20:
        scatter_plot(net.output, y_test)
# =============================================================================
#         scatter_epoch.append([net.output, y_test, epoch])
# =============================================================================
    elif epoch == 70:
        scatter_plot(net.output, y_test)
# =============================================================================
#         scatter_plot(i[0], i[1])
#         scatter_epoch.append([net.output, y_test, epoch])
# =============================================================================
    test_loss.append(loss.item())
    test_acc.append(100 * test_correct/ test_total)
    #print('Test Accuracy:',100 * test_correct/ test_total)

plot(train_loss, train_acc, test_loss, test_acc) 

# =============================================================================
# for i in scatter_epoch:
#     print('%dth learning epochs'%i[2])
#     scatter_plot(i[0], i[1])
# =============================================================================
    
    
    

