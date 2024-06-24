import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

from pythonlibs.my_torch_lib import (
    evaluate_history,
    fit,
    show_images_labels,
    torch_seed,
)

warnings.simplefilter("ignore")
plt.rcParams["axes.grid"] = True
np.set_printoptions(suppress=True, precision=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# remove augumentation in train
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

which_data = "data1"
data_dir = os.path.join("coins_data_sin", which_data)


train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")

which_data_ = os.path.join(which_data, "VGG", "no_EXIF")

classes = [
    str(i1) + str(i2) + str(i3) + str(i4)
    for i1 in range(0, 2)
    for i2 in range(0, 2)
    for i3 in range(0, 2)
    for i4 in range(0, 2)
]

train_data = datasets.ImageFolder(train_dir, transform=train_transform)

train_data2 = datasets.ImageFolder(train_dir, transform=train_transform)

test_data = datasets.ImageFolder(test_dir, transform=test_transform)

batch_size = 10

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

train_loader2 = DataLoader(train_data2, batch_size=50, shuffle=True)
test_loader2 = DataLoader(test_data, batch_size=50, shuffle=True)


show_images_labels(test_loader2, classes, None, None, which_data_)

net = models.vgg19_bn(weights=True)


for param in net.parameters():
    param.requires_grad = False

torch_seed()

in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 16)

net.avgpool = nn.Identity()

net = net.to(device)

lr = 0.001

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.classifier[6].parameters(), lr=lr, momentum=0.9)

history = np.zeros((0, 5))

num_epochs = 10
history = fit(
    net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history
)

evaluate_history(history, which_data_)

show_images_labels(test_loader2, classes, net, device, which_data_)
