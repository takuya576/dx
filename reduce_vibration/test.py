import cv2

# 画像の読み込み
image = cv2.imread("/home/sakamoto/dx/data/data4_1case/0010/IMG_0310.JPG")
# コントラストと明るさの変更
alpha = 1.5  # コントラストの倍率（1より大きい値でコントラストが上がる）
beta = 10  # 明るさの調整値（正の値で明るくなる）
adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
# 変更後の画像の表示
cv2.imshow("Original Image", image)
cv2.imshow("convertScaleAbs", adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
