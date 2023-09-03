from PIL import Image,  ImageTk          # Pillow
from PIL.ExifTags import TAGS,  GPSTAGS  # Exifタグ情報
import torch
import pandas as pd
import os
import glob
import numpy as np

result_dir = "/home/sakamoto/dx/result/"
which_data = "data1"

data_dir = os.path.join("coins_data_sin", which_data)


# exif情報のtagを全て取得する
def exif_tags():
    image = Image.open("/home/sakamoto/dx/coins_data_sin/data1/0000/IMG_8923.jpeg")
    exif_dict = image.getexif()
    # exifタグ情報の取得
    pvtag_dict = exif_dict.get_ifd(34665)
    pvtag = sorted(pvtag_dict.items())  # キーでソート。結果はタプルのリスト

    exif = {}
    for k, v in pvtag:
        if k in TAGS:
            exif[TAGS[k]] = v
    tags = exif.keys()
    return tags


# 画像ファイルからexif情報を抽出し、csvに
def mk_exif_csv(data_dir):
    image_files = glob.glob(os.path.join(data_dir, "[0-9][0-9][0-9][0-9]/*.jpeg"))
    # image_files = glob.glob(os.path.join(data_dir, "0000/IMG_8923.jpeg"))
    tags = exif_tags()
    df_exif = pd.DataFrame(columns=tags)
    df_exif.insert(0, 'File_name', np.NaN)
    for file in image_files:
        # Exif情報の取得
        image = Image.open(file)
        exif_dict = image.getexif()
        # exifタグ情報の取得
        pvtag_dict = exif_dict.get_ifd(34665)
        pvtag = sorted(pvtag_dict.items())  # キーでソート。結果はタプルのリスト

        exif = {}
        for k, v in pvtag:
            if k in TAGS:
                exif[TAGS[k]] = v
        # add_exif = pd.DataFrame(exif)
        add_exif = pd.DataFrame.from_dict(exif, orient='index').transpose()
        add_exif.insert(0, 'File_name', file)
        # add_exif.insert(0, 'File_name', os.path.basename(file))
        df_exif = pd.concat([df_exif, add_exif], ignore_index=True)
        # df_exif.drop_duplicates(inplace=True)
        df_exif.reset_index(drop=True, inplace=True)
    df_exif.to_csv(os.path.join(result_dir, which_data, "EXIF.csv"), index=False)


mk_exif_csv(data_dir)
