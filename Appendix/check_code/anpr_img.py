import cv2
import pytesseract
import numpy as np
import time
import torch
from PIL import Image
# import jetson_inference, jetson_utils
# from skimage.segmentation import clear_border
from anpr_functions import yolo_detector


# net = jetson_inference.detectNet(network="ssd-mobilenet-v2",threshold=0.5)
model =  torch.hub.load('./yolov5', 'custom', source ='local', path='best.pt')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

img = cv2.imread("./images/img1.jpeg")
cv2.imshow('img',img)
cv2.waitKey(0)
img_cropped = cv2.resize(img,(640,480), interpolation = cv2.INTER_AREA)
    
plate = yolo_detector(model,img_cropped)
print(plate)

# dt = time.time()-timestamp
# timestamp=time.time()
# fps=1/dt
# fpsfilt = 0.9*fpsfilt+0.1*fps        
# cv2.putText(img, str(round(fpsfilt,1))+' fps',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    
#cv2.imshow('img',img)

# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
#cam.release()
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
