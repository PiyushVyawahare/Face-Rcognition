import numpy as np
import cv2

image = cv2.imread('goku.jpg')
image = cv2.resize(image, None, fx=0.5, fy=0.5)
height, width, channels = image.shape

M = np.float32([[1,0,50], [0,1,50]])
rotation_matrix = cv2.getRotationMatrix2D((height/2, width/2), 90, 1)

translated_image = cv2.warpAffine(image, M, (height, width))
rotated_image = cv2.warpAffine(image, rotation_matrix, (height, width))
scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5)

cv2.imshow("Image", image)
cv2.imshow("Translated_image", translated_image)
cv2.imshow("Rotated_image", rotated_image)
cv2.imshow("Scaled_image", scaled_image)
cv2.waitKey(0)