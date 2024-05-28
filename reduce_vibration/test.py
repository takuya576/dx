import cv2
import numpy as np

# 画像を読み込む
image = cv2.imread("~/dx/data/data6/data6_1case/1010/0111-130259-3.jpg")
# コントラストと明るさの変更
alpha = 0.1  # コントラストの倍率（1より大きい値でコントラストが上がる）
beta = 20  # 明るさの調整値（正の値で明るくなる）
darkened_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
cv2.imwrite("darkened_image.jpg", darkened_image)

# 画像Aと画像Bの横幅を取得
width = image.shape[1]
print(width)

# 合成したい新しい画像を作成
result = np.zeros_like(image)

# 画像Aと画像Bを指定の比率で合成
for x in range(width):
    a = x / width
    b = 1 - a
    result[:, x] = (image[:, x] * a + darkened_image[:, x] * b).astype(np.uint8)

# 合成された画像を保存または表示
cv2.imwrite("result_linear.jpg", result)

# 変更後の画像の表示
# cv2.imshow("Original Image", image)
# cv2.imshow("convertScaleAbs", adjusted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
