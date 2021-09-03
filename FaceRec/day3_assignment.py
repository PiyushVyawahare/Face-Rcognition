import cv2
cap = cv2.VideoCapture(0)
image1 = cv2.imread("background5.jpg")
image2 = cv2.imread("background.jpg")
image3 = cv2.imread("background1.jpg")
image4 = cv2.imread("background2.jpg")
image5 = cv2.imread("background3.jpg")
image6 = cv2.imread("background4.jpg")
while True:
    flag, frame = cap.read()
    if not flag:
        print("Couldn't read image")
        break
    
    image1 = cv2.resize(image1, (frame.shape[1], frame.shape[0]))
    image2 = cv2.resize(image2, (frame.shape[1], frame.shape[0]))
    image3 = cv2.resize(image3, (frame.shape[1], frame.shape[0]))
    image4 = cv2.resize(image4, (frame.shape[1], frame.shape[0]))
    image5 = cv2.resize(image5, (frame.shape[1], frame.shape[0]))
    image6 = cv2.resize(image6, (frame.shape[1], frame.shape[0]))
    if cv2.waitKey(10) & 0xff == ord('1'):
        image = image1
    if cv2.waitKey(10) & 0xff == ord('2'):
        image1 = image2
    if cv2.waitKey(10) & 0xff == ord('3'):
        image1 = image3
    if cv2.waitKey(10) & 0xff == ord('4'):
        image1 = image4
    if cv2.waitKey(10) & 0xff == ord('5'):
        image1 = image5
    if cv2.waitKey(10) & 0xff == ord('6'):
        image1 = image6
    blended_image = cv2.addWeighted(frame, 0.5, image1, 0.5, 0.1)
    cv2.imshow("Blended Image", blended_image)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()