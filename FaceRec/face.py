import cv2
import face_recognition

image_train = face_recognition.load_image_file("chris.jpg")
image_location_train = face_recognition.face_locations(image_train)[0]
image_encodings_train = face_recognition.face_encodings(image_train)[0]

image_test = face_recognition.load_image_file("chris.png")
image_encodings_test = face_recognition.face_encodings(image_test)[0]
print(image_encodings_train)
results = face_recognition.compare_faces([image_encodings_train], image_encodings_test)[0]
dst = face_recognition.face_distance([image_encodings_train], image_encodings_test)
if results:
    image_train = cv2.cvtColor(image_train, cv2.COLOR_BGR2RGB)
    cv2.rectangle(image_train, (image_location_train[3],image_location_train[0]),(image_location_train[1],image_location_train[2]), (0, 255, 0), 1)
    cv2.putText(image_train, f"{results} {dst}", 
    (20,20),
    cv2.FONT_HERSHEY_DUPLEX,
    1,
    (0, 255, 0),
    1)
    cv2.imshow("Chris", image_train)
else:
    print(f"Couldn't recognise face. Result was {results} and distance was {dst}")
cv2.waitKey(0)
