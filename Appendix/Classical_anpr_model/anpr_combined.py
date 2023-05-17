import cv2
import numpy as np
import time
import torch
from anpr_functions import camset, haarcascade_detector, yolo_detector, put_FPS, put_Rect, put_Text


cam = camset()

# model =  torch.hub.load('./yolov5', 'custom', source ='local', path='best4.pt')
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)


while True:
    ret,frame = cam.read()
    if not ret:
        print("Error retrieving frame")
        break

    # img = cv2.resize(img,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
    # img_cropped = cv2.resize(img,(640,480), interpolation = cv2.INTER_AREA)

    # cropped_img = vehicle_detector(net,img)

    plate,coords = haarcascade_detector(frame)
    top=coords[0]
    left=coords[1]
    bottom=coords[2]
    right=coords[3]

    # plate = yolo_detector(model,img_cropped)
    
    frame,fps = put_FPS(frame)
    frame = put_Rect(frame,top,left,bottom,right)
    frame = put_Text(frame,'License plate',left,bottom,font_scale=0.5,color=(0,255,0))
    cv2.imshow('Haarcascade_plate_detector',frame)
    # cv2.imshow('plate',plate)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()