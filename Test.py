import numpy as np
import cv2 as cv
import time

# OpenCV Facial Capture Test 
landmark_model_path = "C:\\Users\\Joe\\Documents\\AnimationUsingPython\\data\\lbfmodel.yaml"
_cap = cv.VideoCapture(0)
_cap.set(cv.CAP_PROP_FRAME_WIDTH, 512)
_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 512)
_cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
time.sleep(0.5)

facemark = cv.face.createFacemarkLBF()

# error detection 
try:
    # Download the trained model lbfmodel.yaml:
    # https://github.com/kurnianggoro/GSOC2017/tree/master/data
    # and update this path to the file:
    facemark.loadModel(landmark_model_path)
except cv.error:
    print("Model not found")

cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")
if cascade.empty() :
    print("cascade not found")
    exit()

print("Press ESC to stop")

# finite loop 
while True:
    _, frame = _cap.read()
    
    faces = cascade.detectMultiScale(frame, 1.05,  6, cv.CASCADE_SCALE_IMAGE, (130, 130))
    
    #find biggest face, and only keep it
    if(type(faces) is np.ndarray and faces.size > 0):
        biggestFace = np.zeros(shape=(1,4))
        for face in faces:
            if face[2] > biggestFace[0][2]:
                biggestFace[0] = face

        # find landmarks
        ok, landmarks = facemark.fit(frame, faces=biggestFace)
        
        # draw landmarks
        for marks in landmarks:
            for (x, y) in marks[0]:
                cv.circle(frame, (x, y), 2, (0, 255, 255), -1)  
        
        # draw detected face
        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)

        for i,(x,y,w,h) in enumerate(faces):
            cv.putText(frame, "Face #{}".format(i), (x - 10, y - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv.imshow("Image Landmarks", frame)
    if(cv.waitKey(1) == 27):
        exit()

