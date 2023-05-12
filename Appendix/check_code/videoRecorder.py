import cv2

#camera_url = "rtsp://admin:Dd22864549*@10.13.1.60:554/cam/realmonitor?channel=1&subtype=0"
#camera_url = "rtsp://admin:Dd22864549*@192.168.100.159:554/cam/realmonitor?channel=1&subtype=0"
camera_url = "rtsp://admin:Dd22864549*@192.168.100.160:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
	print("Error opening RTSP stream.")
	exit()


#getting width and height of the frames
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#initialize the video writer
#fourcc = cv2.VideoWriter_fourcc(*'X264')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (width, height))

while cap.isOpened():
	ret, frame = cap.read()
    
	# Check if the frame was successfully read
	if not ret:
		print("Error retrieving frame.")
		break

	out.write(frame)
	cv2.imshow('frame',frame)
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()

#Video Codecs supported by opencv
#'X264' for H.264(AVC)
#'XVID' for .avi
#'' for MPEG-4
#'MJPG' for MJPEG
