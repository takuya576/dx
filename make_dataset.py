import os
import random
import shutil

random.seed(123)

# os.makedirs("./coins_data", exist_ok=True)
dataset_dir = "./coins_data/data2"

class_list = [
    str(i1) + str(i2) + str(i3) + str(i4)
    for i1 in range(0, 2)
    for i2 in range(0, 2)
    for i3 in range(0, 2)
    for i4 in range(0, 2)
]
val_rate = 0.2

os.makedirs(os.path.join(dataset_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "val"), exist_ok=True)

for class_name in class_list:
    os.makedirs(os.path.join(dataset_dir, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "val", class_name), exist_ok=True)

for class_name in class_list:
    # for local
    image_dir = os.path.join(dataset_dir, class_name)
    # for gpu server
    # image_dir = os.path.join(os.path.expanduser('~'), "data", "dataset_coins", class_name)
    image_files = os.listdir(image_dir)
    random.shuffle(image_files)
    num_val = int(len(image_files) * val_rate)

    # for test
    for file in image_files[:num_val]:
        src = os.path.join(image_dir, file)
        dst = os.path.join(dataset_dir, "val", class_name)
        shutil.copy(src, dst)

    # for train
    for file in image_files[num_val:]:
        src = os.path.join(image_dir, file)
        dst = os.path.join(dataset_dir, "train", class_name)
        shutil.copy(src, dst)
