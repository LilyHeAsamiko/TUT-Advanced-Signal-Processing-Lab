# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 16:43:42 2018

@author: asamiko
"""
import cv2
import sys

sascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    smile, frame = video_capture.read()
    gray = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeibors = 5,
            minSize = (30,30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        cv2.smile(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    # Displlay the resulting frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    
    video_capture = cv2.VideoCapture(0)
 