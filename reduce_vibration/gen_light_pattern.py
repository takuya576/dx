import json
import os
import pathlib
import shutil

import cv2
from tqdm import tqdm

from utils.image_processing import ImageProcessing
from utils.load_save import load_config

# configでCNN、ハイパーパラメータや使用するデータを指定
config = load_config(config_path=pathlib.Path("/home/sakamoto/dx/config/config.json"))

input_dir = "/home/sakamoto/dx/data/data6/data6_1case"  # 画像ファイルがあるディレクトリのパス
output_dir = os.path.join(
    "/home/sakamoto/dx/generated_data/data6",
    str(config.alpha)
    + "_"
    + str(config.beta)
    + "_"
    + str(config.a)
    + "_"
    + str(config.sigmoid),
)  # 処理後の画像を保存するディレクトリのパス

if os.path.isdir(output_dir) is True:
    shutil.rmtree(output_dir)
# if os.path.isdir(os.path.join(dataset_dir, "val")) is True:
#     shutil.rmtree(os.path.join(dataset_dir, "val"))
os.makedirs(output_dir, exist_ok=True)
# os.makedirs(os.path.join(dataset_dir, "val"), exist_ok=True)

class_list = [
    str(i1) + str(i2) + str(i3) + str(i4)
    for i1 in range(0, 2)
    for i2 in range(0, 2)
    for i3 in range(0, 2)
    for i4 in range(0, 2)
]

for class_name in class_list:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# ディレクトリ内のすべてのファイルを取得
for class_name in tqdm(class_list):
    image_dir = os.path.join(input_dir, class_name)
    image_out_dir = os.path.join(output_dir, class_name)
    for filename in os.listdir(image_dir):
        if filename.endswith(".JPG") or filename.endswith(
            ".jpg"
        ):  # 拡張子がJPGまたはjpgのファイルを対象とする
            image_path = os.path.join(image_dir, filename)  # 画像ファイルのパスを生成

            # 画像処理のためのオブジェクトを作成
            image_processing = ImageProcessing(
                image_path, config.alpha, config.beta, config.a, config.sigmoid
            )

            # 元画像も含めて保存
            result = image_processing.image
            output_path = os.path.join(image_out_dir, filename)
            cv2.imwrite(output_path, result)

            result = image_processing.except_br()
            output_path = os.path.join(image_out_dir, "ex_br_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.except_bl()
            output_path = os.path.join(image_out_dir, "ex_bl_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.except_tr()
            output_path = os.path.join(image_out_dir, "ex_tr_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.except_tl()
            output_path = os.path.join(image_out_dir, "ex_tl_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_right()
            output_path = os.path.join(image_out_dir, "r_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_left()
            output_path = os.path.join(image_out_dir, "l_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_top()
            output_path = os.path.join(image_out_dir, "t_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_bottom()
            output_path = os.path.join(image_out_dir, "b_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_tr_bl()
            output_path = os.path.join(image_out_dir, "tr_bl_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_tl_br()
            output_path = os.path.join(image_out_dir, "tl_br_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_top_right()
            output_path = os.path.join(image_out_dir, "tr_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_bottom_right()
            output_path = os.path.join(image_out_dir, "br_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_top_left()
            output_path = os.path.join(image_out_dir, "tl_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.lit_bottom_left()
            output_path = os.path.join(image_out_dir, "bl_" + filename)
            cv2.imwrite(output_path, result)

            result = image_processing.darken_image()
            output_path = os.path.join(image_out_dir, "dark_" + filename)
            cv2.imwrite(output_path, result)

print("finish")
