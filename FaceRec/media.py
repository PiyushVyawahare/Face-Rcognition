import cv2
import mediapipe as mp

# Drawing utility
mp_drawing = mp.solutions.drawing_utils

#face Detection utility
mp_face_detect = mp.solutions.face_detection

drawing_spec = mp_drawing.DrawingSpec((255, 0, 0), thickness = 1, circle_radius = 1)
#FaceMash
mp_face_mash = mp.solutions.face_mesh

#model for detcting the face
# model_detection = mp_face_detect.FaceDetection()

#model Facemesh
model_facemesh = mp_face_mash.FaceMesh()
cap = cv2.VideoCapture(0)

while True:
    flag, frame = cap.read()
    if not flag:
        print("Couldn't access camera")
        break
    #################For Detection Only################
    #results = model_detection.process(frame)
    #for landmark in results.detections:
    #    mp_drawing.draw_detection(frame, landmark)
        
    #################For Meshing Only##################
    results = model_facemesh.process(frame)
    for landmark in results.multi_face_landmarks:
        print(landmark)
        mp_drawing.draw_landmarks(
            image = frame,
            landmark_list = landmark,
            connections = mp_face_mash.FACE_CONNECTIONS,
            landmark_drawing_spec = drawing_spec,
            connection_drawing_spec = drawing_spec)
        

    cv2.imshow("Frame", frame)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()