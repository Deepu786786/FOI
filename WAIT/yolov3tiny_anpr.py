import cv2
import numpy as np
import torch
from wait_functions import camset, yolo_detector, put_FPS, put_Rect, put_Text


cam = camset()

model =  torch.hub.load('./yolov5', 'custom', source ='local', path='best4.pt')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


while True:
    ret,frame = cam.read()
    if not ret:
        print("Error retrieving frame")
        break

    detections = yolo_detector(model,frame)

    frame,fps = put_FPS(frame)
    
    for detect in detections:
        coord = detect[0]
        conf = detect[1]
        label = detect[2]

        top = coord[0]
        left = coord[1]
        bottom = coord[2]
        right = coord[3]

        frame = put_Rect(frame,top,left,bottom,right)
        frame = put_Text(frame,str(conf),left,bottom,font_scale=0.5,color=(0,0,255))
        
    
        cv2.imshow('YOLOv3-tiny',frame)

    if len(detections)==0:
        cv2.imshow('YOLOv3-tiny',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()