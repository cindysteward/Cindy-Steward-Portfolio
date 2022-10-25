#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 21:14:37 2022

Source: https://realpython.com/face-recognition-with-python/#installing-opencv
Use in terminal: $ python face_detection_code.py [insert filename.png] haarcascade_frontalface_default.xml

Other useful resources: https://pypi.org/project/face-recognition/

@author: cindysteward
"""

import cv2 #OpenCV, used in machine learning and computer vision, originially in C/C++, uses machine learning algorithms to idenitfy faces.
#bite sized chunks of tasks to recognize a face.

import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the cascade, initialize it with faceCascade, so its loaded into memory.
faceCascade = cv2.CascadeClassifier(cascPath) #cascace is XML file with data to detect faces.

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image and convert to grayscale, because OpenCV operates in it.
faces = faceCascade.detectMultiScale( #general function that detects objects
    gray,
    scaleFactor=1.2, #compensates for faces that appear bigger because they're closer, or vice versa.
    #scale factor has to be set up o a case by case basis
    minNeighbors=5, #how many objects detected
    minSize=(30, 30), #gives size of window
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("{0} cute faces have been found!".format(len(faces)))

# For-loop to draw a rectangle around the faces that are detected
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
