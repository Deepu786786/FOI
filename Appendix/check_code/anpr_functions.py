#This is a function file for anpr project
import cv2
import pytesseract
import numpy as np
import time
import torch
from PIL import Image
#import jetson_inference,jetson_utils
import matplotlib.pyplot as plt
#from skimage.segmentation import clear_border

# img = cv2.imread("Numberplate_images/numberplate_11.jpeg")

def vehicle_detector(net,img): 
    height = img.shape[0]
    width = img.shape[1]		
    
    cudaimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    cudaimg =jetson_utils.cudaFromNumpy(cudaimg)
    
    detections = net.Detect(cudaimg,width,height,overlay='lines,labels,conf')

    for detect in detections:
        ID = detect.ClassID
        item = net.GetClassDesc(ID)
        left=detect.Left
        top=detect.Top
        bottom=detect.Bottom
        right=detect.Right

        if item=='car':
            car_img = img[int(top):int(bottom),int(left):int(right)]
            return car_img

    return img


def yolo_detector(model,img):
    
    frame = img
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    
    results = model(frame)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    
    classes = model.names
    # plot_boxes(results, frame, classes = classes)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels



def plot_boxes(results, frame, classes):
  
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cord[i]
        print(row)
    #     if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
    #         print(f"[INFO] Extracting BBox coordinates. . . ")
    #         x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
    #         text_d = classes[int(labels[i])]
    #         # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

    #         coords = [x1,y1,x2,y2]

    #         #plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)


    #         # if text_d == 'mask':
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
    #         cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
    #         cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)

    #         # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])




    # return frame

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate 


def plate_detector(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    if plate_cascade.empty():
        print("Error: Cascade Classifier file not found or cannot be loaded.")
        exit()

    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(plates) == 0:
        print("Error: Number plate not detected.")
        return img

    for (x,y,w,h) in plates:
        a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a, x+b:x+w-b,:]
        return plate



def plate_detector2(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)

    # Sharpening
    gaussian_blur = cv2.GaussianBlur(gray,(7,7),5)
    sharpen = cv2.addWeighted(gray,3.5,gaussian_blur,-2.5,2)
    # gray = cv2.medianBlur(gray,3)
    # gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #Edge detection
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
    blackhat = cv2.morphologyEx(sharpen, cv2.MORPH_BLACKHAT, rectKern)

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    light = cv2.morphologyEx(gray,cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    scharr = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    scharr = np.absolute(scharr)
    (minVal, maxVal) = (np.min(scharr), np.max(scharr))
    scharr = 255 * ((scharr - minVal) / (maxVal - minVal))
    scharr = scharr.astype("uint8")
    scharr = cv2.GaussianBlur(scharr, (5,5),0)
    scharr = cv2.morphologyEx(scharr, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(scharr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh,None, iterations=3)
    thresh = cv2.dilate(thresh,None, iterations=3)
    thresh2 = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh2 = cv2.dilate(thresh2, None, iterations=4)
    thresh2 = cv2.erode(thresh2, None, iterations=4)


    #Contours detection
    cnts,hierarchy = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    ret,license_plate_contour = plate_Filter(cnts)
    
    if ret == -1:
        print("License plate is not detected")
        return img

    x,y,w,h = cv2.boundingRect(license_plate_contour)

    enlarged_img = cv2.resize(img,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
    licensePlate = enlarged_img[y:y+h, x:x+w]
    licensePlate = cv2.resize(licensePlate,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_AREA)
    return licensePlate



def plate_detector3(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)

    #Noise reduction
    gray = cv2.bilateralFilter(gray,11,17,17)
   
    # Sharpening
    gaussian_blur = cv2.GaussianBlur(gray,(7,7),5)
    sharpen = cv2.addWeighted(gray,3.5,gaussian_blur,-2.5,2)

    #Edge detection
    # edged = cv2.Canny(sharpen, 30,200)
    edged = cv2.Canny(sharpen, 100,200)

    #Finding contours
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(gray, gray, mask=mask)

    # (x,y) = np.where(mask==255)
    # (x1,y1) = (np.min(x),np.min(y))
    # (x2,y2) = (np.max(x), np.max(y))

    # cropped_image = gray[x1:x2+1,y1:y2+1]
    # cv2.imshow("cropped_image",cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imshow('img',img)
    cv2.imshow('gray',gray)
    cv2.imshow('edged', edged)
    cv2.imshow('new_image', new_image)    
    cv2.waitKey(0)
    return img



def plate_Filter(cnts):
    filtered_contours=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        #Contour area	
        platearea = cv2.contourArea(c) 
        if platearea <=1500 or platearea >=5000:
            continue
        
        #extent
        rect_area = w*h
        extent = float(platearea)/rect_area
        if extent <= 0.3:
            continue
        
        #Aspect ratio
        aspect_ratio = w/float(h)
        if aspect_ratio<=1.6 or aspect_ratio>=7:
            continue

        #Rectangularity
        _,_, angle = cv2.minAreaRect(c)
        if angle < -45:
            angle += 90
        rectangularity = platearea/rect_area
        if rectangularity <= 0.35:
            continue
        
        filtered_contours.append(c)

    filtered_contours.sort(key=cv2.contourArea, reverse=True)

    if filtered_contours is None:
        print("No contours detected")
        return -1,cnts[0]

    return 1,filtered_contours[0]



def characterSegmentation(plate):
    grayplate = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    # grayplate = cv2.resize(grayplate, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    
    # Noise reduction
    # bilateral = cv2.bilateralFilter(grayplate, d=9, sigmaColor=75, sigmaSpace=75)
    

    # Sharpening
    gaussian_blur = cv2.GaussianBlur(grayplate,(7,7),10)
    sharpen = cv2.addWeighted(grayplate,3.5,gaussian_blur,-2.5,2)
    

    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)


    # White text thickening
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 2)



    cv2.imshow('grayplate',grayplate)
    cv2.imshow('sharpen',sharpen)
    cv2.imshow('thresh',thresh)
    cv2.imshow("dilation", dilation)

    # # Make borders white
    # img_lp = cv2.resize(image, (333, 75))
    # img_binary_lp[0:3,:] = 255
    # img_binary_lp[:,0:3] = 255
    # img_binary_lp[72:75,:] = 255
    # img_binary_lp[:,330:333] = 255

    cv2.waitKey(0)


    # # find contours
    # # try:
    # #     contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # except:
    # #     ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])


    # gplate = grayplate.copy()
    # cv2.drawContours(gplate,contours,-1,(0,255,0),3)


    # # loop through contours and find letters in license plate
    # gplate2 = grayplate.copy()
    # plate_num = ""
    # for cnt in sorted_contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
        
    #     # height, width = gplate2.shape
    #     # # if height of box is not a quarter of total height then skip
    #     # if height / float(h) > 6: continue
    #     # ratio = h / float(w)
    #     # # if height to width ratio is less than 1.5 skip
    #     # if ratio < 1.5: continue
    #     # area = h * w
    #     # # if width is not more than 25 pixels skip
    #     # if width / float(w) > 15: continue
    #     # # if area is less than 100 pixels skip
    #     # if area < 100: continue


    #     # draw the rectangle
    #     rect = cv2.rectangle(gplate2, (x,y), (x+w, y+h), (0,255,0),2)
    #     roi = thresh[y-5:y+h+5, x-5:x+w+5]
    #     roi = cv2.bitwise_not(roi)
    #     roi = cv2.medianBlur(roi, 5)
    #     cv2.imshow("ROI", roi)
    #     try:
    #         text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    #         #clean tesseract text by removing any unwanted blank spaces
    #         clean_text = re.sub('[\W_]+','',text)
    #         plate_num += text
    #     except:
    #         text = None
    # if plate_num != None:
    #     print("License Plate #:",plate_num)




def getNumberPlate(plate):
    #PSM(Page Segmentation Method) mode, Tesseract's setting has 14(0-13) modes of operation, 
    # psm 7 - treat the image as single text line
    # psm 8 - treat the image as a single word
    # psm 10 - treat the image as a single character
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(plate,config='-c tessedit_char_whitelist='+alphanumeric+' --psm 7 --oem 3')
    return text

