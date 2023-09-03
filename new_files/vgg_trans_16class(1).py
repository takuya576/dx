import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
from torch import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

import warnings
warnings.simplefilter('ignore')
plt.rcParams['axes.grid'] = True
np.set_printoptions(suppress=True, precision=5)


from pythonlibs.my_torch_lib import *

import os
import sys

args = sys.argv
program_name = args[0].split('.')[0]
trial = int(args[1])
batch_size = int(args[2])
device = torch.device(f"cuda:{int(args[3])}" if torch.cuda.is_available() else "cpu")


test_transform = transforms.Compose([
    # transforms.Resize(256),
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

#remove augumentation in train
train_transform = transforms.Compose([
    # transforms.Resize(256),
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

#for local
# data_dir = 'coins_data'
#for gpu server
data_dir = "../data/coins_data_beta"

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

classes = [str(i1)+str(i2)+str(i3)+str(i4) for i1 in range(0,2) for i2 in range(0,2) for i3 in range(0,2) for i4 in range(0,2)]

train_data = datasets.ImageFolder(train_dir,
            transform=train_transform)

train_data2 = datasets.ImageFolder(train_dir,
            transform=test_transform)

test_data = datasets.ImageFolder(test_dir,
            transform =test_transform)

#check augumented pictures
# plt.figure(figsize=(15, 4))
# for i in range(10):
#     ax = plt.subplot(2, 10, i + 1)
#     image, label = test_data[i]
#     img = (np.transpose(image.numpy(), (1,2,0)) + 1) / 2
#     plt.imshow(img)
#     ax.set_title(classes[label])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     ax = plt.subplot(2, 10, i + 11)
#     image, label = test_data[-i-1]
#     img = (np.transpose(image.numpy(), (1,2,0)) + 1) / 2
#     plt.imshow(img)
#     ax.set_title(classes[label])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


train_loader = DataLoader(train_data,
    batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_data,
    batch_size=batch_size, shuffle=False)

train_loader2 = DataLoader(train_data2,
    batch_size=50, shuffle=True)
test_loader2 = DataLoader(test_data,
    batch_size=50, shuffle=True)

from torchvision import models
net = models.vgg19_bn(pretrained = True)

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

history = np.zeros((0, 9))

num_epochs = 100

os.makedirs(os.path.join(os.path.expanduser('~'), "static", f"{program_name}_b{trial}"), exist_ok=True)

os.makedirs(os.path.join("results", "beta", f"{program_name}", f"trial{trial}", "confusion_matrix"), exist_ok=True)
os.makedirs(os.path.join("results", "beta", f"{program_name}", f"trial{trial}", "latent_space"), exist_ok=True)

history = fit(net, optimizer, criterion, num_epochs,
        train_loader, test_loader, device, history, program_name, trial)


evaluate_history(history, program_name, trial)

show_images_labels(test_loader2, classes, net, device, program_name, trial)
