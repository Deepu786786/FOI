import torch
import cv2
import time
import pytesseract
import numpy as np
from skimage.segmentation import clear_border



model =  torch.hub.load('./yolov5', 'custom', source ='local', path='best4.pt')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
classes = model.names

def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    print("result: ",results)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates


def getNumberPlate(plate):
    #PSM(Page Segmentation Method) mode, Tesseract's setting has 14(0-13) modes of operation, 
    # psm 7 - treat the image as single text line
    # psm 8 - treat the image as a single word
    # psm 10 - treat the image as a single character
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(plate,config='-c tessedit_char_whitelist='+alphanumeric+' --psm 7 --oem 3')
    return text

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    labels, cord = results
    n = len(labels)

    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    plate = frame
    ### looping through the detections
    for i in range(n):
        row = cord[i]
        
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            coords = [x1,y1,x2,y2]

            #plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)


            # if text_d == 'mask':
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            #cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
       
            plate = frame[y1:y2,x1:x2,:]
       
    return frame,plate


def printNumber(results, frame,classes,text="null"):

    labels, cord = results
    n = len(labels)

    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cord[i]
        
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value

            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates

            coords = [x1,y1,x2,y2]

            frame = put_Text(frame,text,x1,y2,0.5,(0,255,0),2)
       
    return frame



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

	return frame


def camSet():
	camera_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via switch
	#camera_url = "rtsp://admin:Dd22864549*@192.168.100.159:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 4MP camera when connected via 4G routertype `GstRTSPSrc' does not have property `buffer-size'

	#camera_url = "rtsp://admin:Dd22864549*@192.168.100.160:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via 4G router

	# cam = cv2.VideoCapture("rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=1 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER)
	cam = cv2.VideoCapture("/dev/video0")
	
	cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

	if not cam.isOpened():
		print("Error opening RTSP stream.")
		exit()
	
	return cam
cam = camSet()


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


    thresh_cb = clear_border(thresh)
    # cv2.imshow('thresh_cb',thresh_cb)

    thresh_dilatecb = clear_border(thresh_dilate)
    # cv2.imshow('thresh_dilatecb',thresh_dilatecb)
    
    # thresh_cbinv = cv2.bitwise_not(thresh_cb)
    # cv2.imshow('thresh_cvinv',thresh_cbinv)

    thresh_dilatecbinv = cv2.bitwise_not(thresh_dilatecb)
    ##cv2.imshow('thresh_dilatecvinv',thresh_dilatecbinv)

    return thresh_dilatecbinv


def characterSegmentation2(plate):
    grayplate = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    # grayplate = cv2.resize(grayplate, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    
    # Noise reduction
    # bilateral = cv2.bilateralFilter(grayplate, d=9, sigmaColor=75, sigmaSpace=75)
    

    # Sharpening
    gaussian_blur = cv2.GaussianBlur(grayplate,(7,7),10)
    sharpen = cv2.addWeighted(grayplate,3.5,gaussian_blur,-2.5,2)
    

    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)



    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    # White thickening
    thresh_dilate = cv2.dilate(thresh, rect_kern, iterations = 1)


    # Black thickening
    thresh_erode = cv2.erode(thresh,rect_kern, iterations = 1)
    
    
    # cv2.imshow('grayplate',grayplate)
    # cv2.imshow('sharpen',sharpen)
    # cv2.imshow('thresh',thresh)
    # cv2.imshow('thresh_dilate',thresh_dilate)
    # cv2.imshow('thresh_erode',thresh_erode)


    thresh_cb = clear_border(thresh)
    # cv2.imshow('thresh_cb',thresh_cb)

    thresh_dilatecb = clear_border(thresh_dilate)
    # cv2.imshow('thresh_dilatecb',thresh_dilatecb)

    # thresh_erodecb = clear_border(thresh_erode)
    # cv2.imshow('thresh_erodecb',thresh_erodecb)
    

    # thresh_cbinv = cv2.bitwise_not(thresh_cb)
    # cv2.imshow('thresh_cvinv',thresh_cbinv)

    thresh_dilatecbinv = cv2.bitwise_not(thresh_dilatecb)
    cv2.imshow('thresh_dilatecvinv',thresh_dilatecbinv)


    print(getNumberPlate(thresh_dilatecbinv))


    # # Make borders white
    # thresh_wb = cv2.resize(thresh, (333, 75))
    # thresh_wb[0:3,:] = 255
    # thresh_wb[:,0:3] = 255
    # thresh_wb[72:75,:] = 255
    # thresh_wb[:,330:333] = 255

    # cv2.imshow('thresh_wb',thresh_wb)
    # thresh_wbcb = clear_border(thresh_wb)

    # cv2.imshow('img_lpcb',img_lpcb)

    # print(getNumberPlate(img_lpcb))

    # print(pytesseract.image_to_boxes(img_lpcb))



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


def reorient(frame,angle):
    h,w,c =  frame.shape
    center = (h/2,w/2)

    rotation_matrix = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(frame,rotation_matrix,(w,h))



while True:
    ret,frame = cam.read()
    
    if ret:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        # cv2.imshow('frame',frame)

        frame = reorient(frame,angle=15)

        # cv2.imshow('frame_rotated',frame)
        frame,plate = plot_boxes(results, frame,classes = classes)

        ##cv2.imshow("numberplate",plate)

        # frame = put_FPS(frame)
        # cv2.imshow("Camera Feed", frame)

        # print(getNumberPlate(plate))
        segmented_plate = characterSegmentation(plate)
        number = getNumberPlate(segmented_plate)

        frame = printNumber(results,frame,classes=classes,text=number)

        frame = put_FPS(frame)
        cv2.imshow("Camera Feed", frame)


    if not ret:
        print("Error retrieving frame")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break        

cam.release()
cv2.destroyAllWindows()
