import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

cap = cv2.VideoCapture(0)
dm_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=dm_factor, fy=dm_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.1, 20)
    for (x,y,w,h) in mouth_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
        break

    cv2.imshow('Nose Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()