import pathlib
import cv2
import numpy as np

class ImageProcessing(pathlib.Path:path):
    def __init__(self) -> None:
        image = cv2.imread(path)

    def darken_image(self) -> np.ndarray:
