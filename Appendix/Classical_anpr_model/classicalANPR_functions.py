import cv2
import jetson_inference,jetson_utils
import pytesseract
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border



def camset():

    # camera_url = "rtsp://admin:Dd22864549*@10.13.1.60:554/cam/realmonitor?channel=1&subtype=0"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=480, height=360 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=640, height=480 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=1280, height=720 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=1920, height=1080 ! appsink drop=1"
    
    # cam = cv2.VideoCapture(camera_url, cv2.CAP_GSTREAMER)

    cam = cv2.VideoCapture('/dev/video0')
    # cam.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    if not cam.isOpened():
        print("Error opening camera.")

    return cam


def opencvToCuda(frame):
    cudaimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    cudaimg = jetson_utils.cudaFromNumpy(cudaimg)
    return cudaimg


def cudaToOpencv(cudaimg,width,height):

    numpy_array = jetson_utils.cudaToNumpy(cudaimg,width,height,4)
    img = cv2.cvtColor(numpy_array.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    return img


def put_Text(frame,text='NoText', x=10, y=10, font_scale=2, color=(0,0,255), text_thickness=1):

	if isinstance(text,float) or isinstance(text,int):
		text = str(round(text,2))
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	text_size = cv2.getTextSize(text,font, font_scale, text_thickness)[0]
	text_x = x + 10
	text_y = y + 15

	return cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, text_thickness)


timestamp = time.time()
fpsfilt=0
def put_FPS(frame):
	global timestamp, fpsfilt
	dt = time.time()-timestamp
	timestamp=time.time()
	fps=1/dt
	fpsfilt = 0.9*fpsfilt+0.1*fps
	
	text = 'FPS: '+str(round(fpsfilt,2))
	frame = put_Text(frame,text,x=5,y=10,font_scale=1,text_thickness=2)

	return frame,text


def put_Rect(img,top,left,bottom,right):
	green_color = (0,255,0)
	thickness = 1
	start_point = (left,top)
	end_point = (right,bottom)

	img = cv2.rectangle(img, start_point, end_point, green_color, thickness)

	return img



def vehicle_detector(net,img): 
    height,width = img.shape[0],img.shape[1]		
    
    cudaimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    cudaimg =jetson_utils.cudaFromNumpy(cudaimg)
    
    detections = net.Detect(cudaimg,width,height)

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

    # if filtered_contours is None:
    if len(filtered_contours) == 0:
        print("No contours detected")
        return -1,cnts[0]

    return 1,filtered_contours[0]



def plateDetector_sobel(img):
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



def plateDetector_canny(img):
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



def characterSegmentation(plate):
    grayplate = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)

    # Sharpening
    gaussian_blur = cv2.GaussianBlur(grayplate,(7,7),10)
    sharpen = cv2.addWeighted(grayplate,3.5,gaussian_blur,-2.5,2)

    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    # White thickening
    thresh_dilate = cv2.dilate(thresh, rect_kern, iterations = 1)
    thresh_dilatecb = clear_border(thresh_dilate)

    # thresh_dilatecbinv = cv2.bitwise_not(thresh_dilate)
    thresh_dilatecbinv = cv2.bitwise_not(thresh_dilatecb)

    return thresh_dilatecbinv



def getNumberPlate(plate):
    #pytesseract perform better when reading black text on white background because tesseract's underlying algorithms are generally optimized for such scenarios.
    #PSM(Page Segmentation Method) mode, Tesseract's setting has 14(0-13) modes of operation, 
    # psm 7 - treat the image as single text line
    # psm 8 - treat the image as a single word
    # psm 10 - treat the image as a single character
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(plate,config='-c tessedit_char_whitelist='+alphanumeric+' --psm 7 --oem 3')
    return text

