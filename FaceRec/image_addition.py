import cv2

image1 = cv2.imread("goku.jpg")
image2 = cv2.imread("background.jpg")
image2 = cv2.resize(image2, (image1.shape[1], image2.shape[0]+406))
added_image = image1 + image2
blended_image = cv2.addWeighted(image1, 0.7, image2, 0.3, 0.1)

cv2.imshow("Original Image", image1)
cv2.imshow("Added Image", added_image)
cv2.imshow("Blended Image", blended_image)

cv2.waitKey(0)