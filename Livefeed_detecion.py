import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
import h5py

mask_detection=load_model('mask_detection.h5')

faceCascade = cv2.CascadeClassifier('E:/vscode/xmls/haarcascade_frontalface_alt2.xml')

video_capture = cv2.VideoCapture(0)
while True: #(or 1)
    # Capture frame-by-frame
    
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame,
                                         scaleFactor=1.02,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in faces:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite('temp.jpg', face_frame)
        X = image.load_img('temp.jpg', target_size=(150,150,3))
        X = image.img_to_array(X)
        Z = np.expand_dims(X,axis=0)
        val1 = mask_detection.predict(Z)
        
        if int(val1) == 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 3)
            cv2.putText(frame,'Mask',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        if int(val1) == 1:
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,255), 3)
            cv2.putText(frame,'No mask',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()