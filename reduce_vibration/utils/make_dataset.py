import os
import random
import shutil


def make_dataset(dataset_dir, val_rate):
    random.seed(123)

    class_list = [
        str(i1) + str(i2) + str(i3) + str(i4)
        for i1 in range(0, 2)
        for i2 in range(0, 2)
        for i3 in range(0, 2)
        for i4 in range(0, 2)
    ]

    if os.path.isdir(os.path.join(dataset_dir, "train")) is True:
        shutil.rmtree(os.path.join(dataset_dir, "train"))
    if os.path.isdir(os.path.join(dataset_dir, "val")) is True:
        shutil.rmtree(os.path.join(dataset_dir, "val"))
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

        # ファイル名が".DS_Store"で終わるファイルを除外
        image_files = [file for file in image_files if not file.endswith(".DS_Store")]

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
