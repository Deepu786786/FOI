import cv2

camera_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via switch
#camera_url = "rtsp://admin:Dd22864549*@192.168.100.159:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 4MP camera when connected via 4G router
#camera_url = "rtsp://admin:Dd22864549*@192.168.100.160:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via 4G router

cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
	print("Error opening RTSP stream.")
	exit()

while True:
	ret, frame = cap.read()
    
	# Check if the frame was successfully read
	if ret:
		frame = cv2.resize(frame,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
		#frame = cv2.cvtColor(frame,cv2.COLOR_RGB2RGBA)
		cv2.imshow("Camera Feed", frame)
		print(frame.shape)

	if not ret:
		print("Error retrieving frame.")
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()


#8MP camera resolution: 3840*2160
