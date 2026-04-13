import cv2
import numpy as np

cap=cv2.VideoCapture(0)
face_lbp_cascade=cv2.CascadeClassifier('lbpcascade_frontalface.xml') # Downloaded from https://github.com/opencv/opencv/tree/master/data/lbpcascades
while True:
    ret,frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    lbp_faces=face_lbp_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
        )
    for (x,y,w,h) in lbp_faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()