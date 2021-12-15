import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt

def get_current_model():

    return model

NUM_CLASSES = 2

model = nn.Sequential(
    nn.Conv2d(3, 8, (8,8), stride=1,padding=0),
    nn.LeakyReLU(),
    nn.Conv2d(8, 16, (5,5), stride=1,padding=0),
    nn.LeakyReLU(),
    nn.Conv2d(16, 32, (3,3), stride=1,padding=0),
    nn.LeakyReLU(),
    #nn.MaxPool2d((2,2), stride=2),
    nn.Flatten(),
    nn.Linear(11552, 1000),
    nn.LeakyReLU(),
    nn.Linear(1000, 200),
    nn.LeakyReLU(),
    nn.Linear(200, NUM_CLASSES)
)