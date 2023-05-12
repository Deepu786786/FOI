import cv2
# import jetson_inference
# import jetson.utils
import time
import numpy as np
import torch


print(f"[INFO] Loading model... ")
## loading the custom trained model
#model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
model =  torch.hub.load('./yolov5', 'custom', source ='local', path='best3.pt',force_reload=True) ### The repo is stored locally
classes = model.names ### class names in string format

def put_Rect(img,top,left,bottom,right):
	green_color = (0,255,0)
	thickness = 1
	start_point = (left,top)
	end_point = (right,bottom)

	img = cv2.rectangle(img, start_point, end_point, green_color, thickness)

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

	return frame


def set_Camera():

	camera_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via switch

	# ##Using OpenGL Library
	# disW = 640
	# disH = 480
	# cam = jetson.utils.gstCamera(disW,disH,"/dev/video0")


	# Using OpenCv library
	cam = cv2.VideoCapture(camera_url)
	cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	if not cam.isOpened():
		print("Error opening RTSP stream.")
		exit()

	return cam

def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            # cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            # cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)

            # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])




    return frame

def main(img_path):

    # frame = cv2.imread(img_path)
    frame = img_path
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result = detectx(frame, model = model)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = plot_boxes(result, frame, classes= classes)

    return frame

cam = set_Camera()
		# img = jetson.utils.cudaFromNumpy(img)
while True:
    print("hello")
    ret,frame = cam.read()
    print(ret)
    if ret:

        print("output frame")
        img = main(img_path=frame)
        cv2.imshow("Display",img)
    if not ret:
        print("Error retrieving frame")
        exit()
    if cv2.waitKey(1) &0xFF ==ord("q"):
        break    
        # cv2.imshow("Display")

	# if ret:
    #     # frame = main(img_path=frame)
    #     print("hello")

	# 	print("output_frame")
	# 	cv2.imshow("Display", frame)


	# if not ret:
	# 	print("Error retrieving frame.")
	# 	#exit()
	
	# if cv2.waitKey(1) & 0xFF == ord("q"):
	# 	break

cam.release()
cv2.destroyAllWindows()


