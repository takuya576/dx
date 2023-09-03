from PIL import Image,  ImageTk          # Pillow
from PIL.ExifTags import TAGS,  GPSTAGS  # Exifタグ情報
import torch


def extract_exif(sample):
    # Exif情報の取得
    exif_dict = sample.getexif()

    # exifタグ情報の取得
    pvtag_dict = exif_dict.get_ifd(34665)
    pvtag = sorted(pvtag_dict.items())  # キーでソート。結果はタプルのリスト

    exif = {}
    for k, v in pvtag:
        if k in TAGS:
            exif[TAGS[k]] = v

    # return normalized values
    return torch.Tensor([exif["ExposureTime"], exif["FNumber"]/3.2, exif["ISOSpeedRatings"]/1000])
    # return torch.Tensor([exif["ExposureTime"]])

