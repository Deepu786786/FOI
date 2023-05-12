import cv2
import jetson.inference
import jetson.utils
import time
import numpy as np

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

	camera_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0"
	# cam = cv2.VideoCapture(camera_url)
	cam = cv2.VideoCapture('/dev/video0')

	cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	if not cam.isOpened():
		print("Error opening RTSP stream.")
		exit()

	return cam



net = jetson_inference.detectNet(network="ssd-mobilenet-v2",threshold=0.5)
cam = set_Camera()
		


while True:
	ret, frame = cam.read()

	if ret:
		height,width = frame.shape[0],frame.shape[1]	
		img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
		img = jetson.utils.cudaFromNumpy(img)
		
		detections = net.Detect(img,width,height)

		vehicleCount=0
		carcount=0
		for detect in detections:
			ID = detect.ClassID
			item = net.GetClassDesc(ID)
			conf = detect.Confidence
			top, bottom, left, right = int(detect.Top), int(detect.Bottom), int(detect.Left), int(detect.Right)

			if ID>1 and ID<10:
				vehicleCount+=1
			if item=='car':
				carcount+=1

			frame = put_Rect(frame,top,left,bottom,right)
			frame = put_Text(frame, item, x=left-5, y=top, font_scale=0.5, color=(0,255,0), text_thickness=1)
			text = 'Confidence: ' + str(round(conf,4)*100) + '%'
			frame = put_Text(frame, text, x=left-5, y=top+15, font_scale=0.5, color=(0,255,0), text_thickness=1)

		frame = put_FPS(frame)
		frame = put_Text(frame, 'Vehicle Count = '+str(vehicleCount), x=200, y=10, font_scale=1, text_thickness=2)
		cv2.imshow("Display", frame)


	if not ret:
		print("Error retrieving frame.")
		#exit()
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cam.release()
cv2.destroyAllWindows()

#To increase kernel socket buffer max size on receiver side(jetson)
#sudo sysctl -w net.core.rmem_max=26214400





