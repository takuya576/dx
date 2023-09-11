import os
import sys
import time
from datetime import datetime

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

# 開始時間を記録
start_time = time.time()

args = sys.argv
program_name = args[0].split(".")[0]
batch_size = int(args[1])
device = torch.device(f"cuda:{int(args[2])}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

which_data = "data4_all_cases"
data_dir = os.path.join("/home/sakamoto/dx/coins_data", which_data)

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")

# Get the current date and time
now = datetime.now()
Date = now.strftime("%Y-%m-%d")
Time = now.strftime("%H-%M-%S")

# Create the directory name
when = f"{Date}_{Time}"

save_dir = os.path.join(
    "/home/sakamoto/dx/result", which_data, f"{when}_{program_name}"
)
os.makedirs(save_dir, exist_ok=True)


test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

# remove augumentation in train
train_transform = transforms.Compose(
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

train_data = datasets.ImageFolder(train_dir, transform=train_transform)

train_data2 = datasets.ImageFolder(train_dir, transform=test_transform)

test_data = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(
    train_data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True
)

test_loader = DataLoader(
    test_data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False
)

train_loader2 = DataLoader(
    train_data2, batch_size=50, num_workers=2, pin_memory=True, shuffle=True
)
test_loader2 = DataLoader(
    test_data, batch_size=50, num_workers=2, pin_memory=True, shuffle=True
)

net = models.resnet18(pretrained=True)

torch_seed()

fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, 16)

net = net.to(device)

lr = 0.001

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

history = np.zeros((0, 9))

num_epochs = 100

history = fit(
    net,
    optimizer,
    criterion,
    num_epochs,
    train_loader,
    test_loader,
    device,
    history,
    program_name,
    save_dir,
    which_data,
)


evaluate_history(history, program_name, save_dir)

show_images_labels(test_loader2, classes, net, device, program_name, save_dir)

# 終了時間を記録
end_time = time.time()

# 実行時間を計算して表示
execution_time = end_time - start_time
# ファイルを開く
with open(f"{save_dir}/{program_name}_abst.txt", "a") as f:
    # ファイルに出力する
    print("実行時間:", execution_time, "秒", file=f)
