import os
import pathlib
import shutil
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
from torch.utils.data import DataLoader

from pythonlibs.my_torch_lib import (
    evaluate_history,
    fit,
    show_images_labels,
    torch_seed,
)
from utils.const import model_mapping
from utils.count_files import count_JPG_files
from utils.load_save import load_config
from utils.make_dataset import make_dataset

plt.rcParams["font.size"] = 18
plt.tight_layout()

config = load_config(config_path=pathlib.Path("~/dx/config/config.json"))

# dataset_dir = "./data/data4_all_cases"

# val_rate = config.num_val / 16

# make_dataset(dataset_dir, val_rate)

# 開始時間を記録
start_time = time.time()

args = sys.argv
program_name = args[0].split(".")[0].split("/")[-1]
batch_size = config.batch_size
device = torch.device(
    f"cuda:{int(config.nvidia)}" if torch.cuda.is_available() else "cpu"
)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

which_data = config.which_data

train1_dir = os.path.join("~/dx/data", config.train_data_1)
train2_dir = os.path.join("~/dx/data", config.train_data_2)
test_dir = os.path.join("~/dx/data", config.test_data)

# Get the current date and time
now = datetime.now()
Date = now.strftime("%Y-%m-%d")
Time = now.strftime("%H-%M-%S")

# Create the directory name
when = f"{Date}_{Time}"

save_dir = os.path.join("~/dx/result", which_data, when)
os.makedirs(save_dir, exist_ok=True)

# 実行時jsonを保存する
shutil.copy(src="~/dx/config/config.json", dst=save_dir)


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

train1_data = datasets.ImageFolder(train1_dir, transform=test_transform)

train2_data = datasets.ImageFolder(train2_dir, transform=test_transform)

test_data = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader1 = DataLoader(
    train1_data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True
)

train_loader2 = DataLoader(
    train2_data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True
)

test_loader1 = DataLoader(
    test_data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False
)

test_loader = DataLoader(
    test_data, batch_size=50, num_workers=2, pin_memory=True, shuffle=True
)

net1 = model_mapping[config.net](pretrained=config.pretrained)
net2 = model_mapping[config.net](pretrained=config.pretrained)

torch_seed()

fc_in_features = net1.fc.in_features
fc_in_features = net2.fc.in_features
net1.fc = nn.Linear(fc_in_features, 16)
net2.fc = nn.Linear(fc_in_features, 16)


net1 = net1.to(device)
net2 = net2.to(device)

criterion = nn.CrossEntropyLoss()

optimizer1 = optim.SGD(net1.parameters(), lr=config.lr, momentum=config.momentum)
optimizer2 = optim.SGD(net2.parameters(), lr=config.lr, momentum=config.momentum)

history1 = np.zeros((0, 9))
history2 = np.zeros((0, 9))

num_data1 = count_JPG_files(train1_dir)
num_data2 = count_JPG_files(train2_dir)


# 学習データが少ない方はその分エポック数を増やす
num_epochs1 = config.num_epochs
num_epochs2 = int(config.num_epochs * (num_data1 / num_data2))

history1 = fit(
    net1,
    optimizer1,
    criterion,
    num_epochs1,
    train_loader1,
    test_loader1,
    device,
    history1,
    save_dir,
    which_data,
    False,
    False,
)

history2 = fit(
    net2,
    optimizer2,
    criterion,
    num_epochs2,
    train_loader2,
    test_loader1,
    device,
    history2,
    save_dir,
    which_data,
    False,
    False,
)

evaluate_history(history1, save_dir, config.train_data_1)
evaluate_history(history2, save_dir, config.train_data_2)

show_images_labels(test_loader, classes, net1, device, config.train_data_1, save_dir)
show_images_labels(test_loader, classes, net2, device, config.train_data_2, save_dir)

# 終了時間を記録
end_time = time.time()

# 実行時間を計算して表示
execution_time = end_time - start_time
# ファイルを開く
with open(f"{save_dir}/abst.txt", "a") as f:
    # ファイルに出力する
    print("実行時間:", execution_time, "秒", file=f)
