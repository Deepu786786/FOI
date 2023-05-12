import cv2
import pytesseract
import numpy as np
import time
import jetson_inference, jetson_utils
from skimage.segmentation import clear_border
from anpr_functions import plate_detector, plate_detector2, plate_detector3, getNumberPlate, characterSegmentation, vehicle_detector




net = jetson_inference.detectNet(network="ssd-mobilenet-v2",threshold=0.5)
camera_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via switch
#camera_url = "rtsp://admin:Dd22864549*@192.168.100.159:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 4MP camera when connected via 4G router
#camera_url = "rtsp://admin:Dd22864549*@192.168.100.160:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via 4G router

cam = cv2.VideoCapture(camera_url)

if not cam.isOpened():
    print("Error opening RTSP stream.")
	#exit()
timestamp = time.time()
fpsfilt=0
while True:
    ret,img=cam.read()
    if not ret:
        print("Error retrieving frame")
        break

    img = cv2.resize(img,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
    #cropped_img = vehicle_detector(net,img)

    #plate = plate_detector2(img)

    cv2.imshow('img',img)
    #cv2.imshow('cropped_img',cropped_img)
    #cv2.imshow('plate',plate)

    # FPS
    dt = time.time()-timestamp
    timestamp=time.time()
    fps=1/dt
    fpsfilt = 0.9*fpsfilt+0.1*fps        
    cv2.putText(img, str(round(fpsfilt,1))+' fps',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()




# img = cv2.imread("Numberplate_images/goodplates/numberplate_2.jpeg")
# # img = cv2.imread("Numberplate_images/numberplate_13.jpeg")	#for plate_detector3

# # net = jetson_inference.detectNet(network="ssd-mobilenet-v2",threshold=0.5)
# # img = vehicle_detector(net,img)



# plate = plate_detector2(img)

# plate = cv2.resize(plate,None,fx=10,fy=10,interpolation=cv2.INTER_CUBIC)
# cv2.imshow('plate',plate)


# characterSegmentation(plate)




# # text = getNumberPlate(plate)
# # print(text)

# cv2.destroyAllWindows()
