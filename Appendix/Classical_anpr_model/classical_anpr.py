import cv2
import numpy as np
from anpr_functions import camset, put_FPS, put_Rect, put_Text, plateDetector_sobel


cam = camset()

while True:
    ret,frame = cam.read()
    if not ret:
        print("Error retrieving frame")
        break

    # frame = cv2.resize(frame,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
    # frame_cropped = cv2.resize(frame,(640,480), interpolation = cv2.INTER_AREA)

    # plate = plateDetector_sobel(frame)
    

    frame,fps = put_FPS(frame)
    
    
    cv2.imshow('YOLOv3-tiny',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

#To increase kernel socket buffer max size on receiver side(jetson)
#sudo sysctl -w net.core.rmem_max=26214400