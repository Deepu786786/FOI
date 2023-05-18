import cv2
import numpy as np
from wait_functions import camset, put_FPS, put_Rect, put_Text, 


cam = camset()

while True:
    ret,frame = cam.read()
    if not ret:
        print("Error retrieving frame")
        break

    plate,coords = haarcascade_detector(frame)
    frame,fps = put_FPS(frame)
    
    cv2.imshow('Haarcascade_Plate_Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()