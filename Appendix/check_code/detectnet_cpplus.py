import cv2
import jetson.inference
import time
import numpy as np

net = jetson.inference.detectNet(network="ssd-mobilenet-v2",threshold=0.5)

camera_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via switch
#camera_url = "rtsp://admin:Dd22864549*@192.168.100.159:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 4MP camera when connected via 4G router
#camera_url = "rtsp://admin:Dd22864549*@192.168.100.160:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via 4G router

cam = cv2.VideoCapture("/dev/video0")

if not cam.isOpened():  
	print("Error opening RTSP stream.")
	exit()

timestamp = time.time()
fpsfilt=0
while True:
	ret, frame = cam.read()
	frame = cv2.resize(frame,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
    
	# Check if the frame was successfully read
	if ret:
		
		height = frame.shape[0]
		width = frame.shape[1]		
		
		img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
		img=jetson.utils.cudaFromNumpy(img)
		
		#detections = net.Detect(img,width,height,overlay='lines,labels,conf')
		detections = net.Detect(img,width,height)
		

		#cv2.imshow("Detections", cv2.UMat(img))
		cv2.imshow("Display", frame)
		
		dt = time.time()-timestamp
		timestamp=time.time()
		fps=1/dt
		fpsfilt = 0.9*fpsfilt+0.1*fps
		#cv2.putText(frame, str(round(fpsfilt,1))+' fps',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
		
		carcount=0
		for detect in detections:
			ID = detect.ClassID
			item = net.GetClassDesc(ID)
			left=detect.Left
			top=detect.Top
			bottom=detect.Bottom
			right=detect.Right
			#print(item,top,left,bottom,right,fpsfilt)
			if item=='car':
				carcount+=1

		print('total cars = ',carcount)

	if not ret:
		print("Error retrieving frame.")
		#exit()
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cam.release()
cv2.destroyAllWindows()


#8MP camera resolution: 3840*2160

#To increase kernel socket buffer max size on receiver side(jetson)
#sudo sysctl -w net.core.rmem_max=26214400



