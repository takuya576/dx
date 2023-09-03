import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models

from pythonlibs.my_torch_lib import (evaluate_history, fit, show_images_labels,
                                     torch_seed)

args = sys.argv
program_name = args[0].split(".")[0]
batch_size = int(args[1])
device = torch.device(f"cuda:{int(args[2])}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

which_data = "data2"
data_dir = os.path.join("coins_data_sin", which_data)

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")


save_dir = "/home/sakamoto/dx"


test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)


classes = [
    str(i1) + str(i2) + str(i3) + str(i4)
    for i1 in range(0, 2)
    for i2 in range(0, 2)
    for i3 in range(0, 2)
    for i4 in range(0, 2)
]

test_data = datasets.ImageFolder(test_dir, transform=test_transform)

# test_loader2 = DataLoader(
#     test_data, batch_size=50, num_workers=2, pin_memory=True, shuffle=True
# )
test_loader2 = DataLoader(
    test_data, batch_size=50, shuffle=True
)

show_images_labels(test_loader2, classes, None, device, program_name, save_dir)

plt.clf()
#画像の読み込み
im = Image.open("/home/sakamoto/dx/coins_data_sin/data2/0000/IMG_9302.JPG")
im = im.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(im)
plt.savefig("sample.png")

plt.clf()
img = plt.imread(r"/home/sakamoto/dx/coins_data_sin/data2/0000/IMG_9302.JPG")
plt.imshow(img, origin='upper')
plt.savefig("sample2.png")