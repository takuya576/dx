import cv2
import numpy as np


class ImageProcessing:
    def __init__(self, image_path, alpha, beta, a) -> None:
        self.image = cv2.imread(image_path)
        self.alpha = alpha  # コントラストの倍率（1より大きい値でコントラストが上がる）
        self.beta = beta  # 明るさの調整値（正の値で明るくなる）
        self.a = a  # シグモイド関数の傾斜を調整するパラメータ

    # ３区画明るくする
    # 右下以外
    def except_br(self, image=None):
        if image is None:
            image = self.image
        lit_left = self.lit_left()
        result = self.synthesize(lit_left, image, horizontal=False)
        return result

    # 左下以外
    def except_bl(self, image=None):
        if image is None:
            image = self.image
        lit_right = self.lit_right()
        result = self.synthesize(lit_right, image, horizontal=False)
        return result

    # 右上以外
    def except_tr(self, image=None):
        if image is None:
            image = self.image
        lit_left = self.lit_left()
        result = self.synthesize(image, lit_left, horizontal=False)
        return result

    # 左上以外
    def except_tl(self, image=None):
        if image is None:
            image = self.image
        lit_right = self.lit_right()
        result = self.synthesize(image, lit_right, horizontal=False)
        return result

    # ２区画明るくする
    # 画像の右を明るくする
    def lit_right(self, image=None):
        if image is None:
            image = self.image
        darkened_image = self.darken_image(image)
        result = self.synthesize(image, darkened_image)
        return result

    # 画像の左を明るくする
    def lit_left(self, image=None):
        if image is None:
            image = self.image
        darkened_image = self.darken_image(image)
        result = self.synthesize(darkened_image, image)
        return result

    # 画像の上を明るくする
    def lit_top(self, image=None):
        if image is None:
            image = self.image
        darkened_image = self.darken_image(image)
        result = self.synthesize(darkened_image, image, horizontal=False)
        return result

    # 画像の下を明るくする
    def lit_bottom(self, image=None):
        if image is None:
            image = self.image
        darkened_image = self.darken_image(image)
        result = self.synthesize(image, darkened_image, horizontal=False)
        return result

    # 右上&左下
    def lit_tr_bl(self, image=None):
        if image is None:
            image = self.image
        lit_top = self.lit_top()
        lit_bottom = self.lit_bottom()
        result = self.synthesize(lit_top, lit_bottom)
        return result

    # 左上&右下
    def lit_tl_br(self, image=None):
        if image is None:
            image = self.image
        lit_top = self.lit_top()
        lit_bottom = self.lit_bottom()
        result = self.synthesize(lit_bottom, lit_top)
        return result

    # １区画明るくする
    # 画像の右上を明るくする
    def lit_top_right(self, image=None):
        if image is None:
            image = self.image
        lit_right = self.lit_right(image)
        darkened_image = self.darken_image(image)
        result = self.synthesize(darkened_image, lit_right, horizontal=False)
        return result

    # 画像の右下を明るくする
    def lit_bottom_right(self, image=None):
        if image is None:
            image = self.image
        lit_right = self.lit_right(image)
        darkened_image = self.darken_image(image)
        result = self.synthesize(lit_right, darkened_image, horizontal=False)
        return result

    # 画像の左上を明るくする
    def lit_top_left(self, image=None):
        if image is None:
            image = self.image
        lit_left = self.lit_left(image)
        darkened_image = self.darken_image(image)
        result = self.synthesize(darkened_image, lit_left, horizontal=False)
        return result

    # 画像の左下を明るくする
    def lit_bottom_left(self, image=None):
        if image is None:
            image = self.image
        lit_left = self.lit_left(image)
        darkened_image = self.darken_image(image)
        result = self.synthesize(lit_left, darkened_image, horizontal=False)
        return result

    # 画像を暗くする
    def darken_image(self, image=None) -> np.ndarray:
        if image is None:
            image = self.image
        darkened_image = cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
        # cv2.imwrite("darkened_image.jpg", darkened_image)
        return darkened_image

    def synthesize(self, image1, image2, horizontal=True, sigmoid=True) -> np.ndarray:
        if sigmoid:
            result = self.sigmoid_synth(image1, image2, horizontal)
        else:
            result = self.linear_synth(image1, image2, horizontal)
        return result

    def sigmoid_synth(self, image1, image2, horizontal=True):
        height = image1.shape[0]
        width = image1.shape[1]
        result = np.zeros_like(image1)
        if horizontal:
            # 水平方向にシグモイド関数を使用して合成
            for x in range(width):
                ratio = 1 / (1 + np.exp(-self.a * (x / width - 0.5)))
                result[:, x] = (
                    image1[:, x] * ratio + image2[:, x] * (1 - ratio)
                ).astype(np.uint8)
        else:
            # 垂直方向にシグモイド関数を使用して合成
            for y in range(height):
                ratio = 1 / (1 + np.exp(-self.a * (y / height - 0.5)))
                result[y, :] = (
                    image1[y, :] * ratio + image2[y, :] * (1 - ratio)
                ).astype(np.uint8)
        return result

    def linear_synth(self, image1, image2, horizontal=True):
        height = image1.shape[0]
        width = image1.shape[1]
        result = np.zeros_like(image1)
        if horizontal:
            # 水平方向に線形関数を使用して合成
            for x in range(width):
                q = x / width
                r = 1 - q
                result[:, x] = (image1[:, x] * q + image2[:, x] * r).astype(np.uint8)
        else:
            # 垂直方向にシグモイド関数を使用して合成
            for y in range(height):
                q = y / height
                r = 1 - q
                result[y, :] = (image1[y, :] * q + image2[y, :] * r).astype(np.uint8)
        return result
