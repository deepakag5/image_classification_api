# %%-----------------------------------------------------------------------
# Loading required packages

import matplotlib.pyplot as plt
import numpy as np  # For storing data as numpy arrays
import pandas as pd
import timeit

# %%-----------------------------------------------------------------------

# importing torch packages
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image  # For handling the images
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


#specify the class for preprocessing the data
class DatasetProcessing(Dataset):
    """
    This function is used to initialise the class variables - transform, data, target

    """

    def __init__(self, data, target, transform=None):
        self.transform = transform
        self.data = data.reshape((-1, 120, 320)).astype(np.uint8)[:, :, :, None]
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.target[index]

    def __len__(self):
        return len(list(self.data))


# specify the model class
class CNN(nn.Module):
    '''
    Here we are using CNN model with three conv layers with maxpool
    and relu as transfer/activation function
    for fully connected layer again we have used relu activation function
    '''

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
