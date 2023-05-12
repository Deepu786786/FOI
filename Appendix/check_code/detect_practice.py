import jetson.inference
import jetson.utils
import cv2

net = jetson.inference.detectNet(network="ssd-mobilenet-v2" ,threshold=0.1)


camera_url = "rtsp://admin:Dd22864549*@10.13.1.60:554/cam/realmonitor?channel=1&subtype=0"
camera = cv2.VideoCapture(camera_url)

#camera = jetson.utils.gstCamera(640,480,"rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0 latency=0 ! rtph264depay ! h264parse ! nvv412decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480")
display = jetson.utils.glDisplay()



while display.IsOpen():

	ret,frame = camera.read()
	if not ret:
		print("Error retrieving frame")
		break
	
	img = cv2.cvtColor(frame,cv2.COLOR_RGB2RGBA)

	img, width, height = camera.CaptureRGBA()
	
	detections = net.Detect(img, width, height, overlay='lines,labels,conf')
	
	vehicleCount=0
	for obj in detections:
		if obj.ClassID>1 and obj.ClassID<10:
			vehicleCount+=1



#	display.RenderOnce(img, width, height)
	#display.BeginRender()
	#display.Render(img)
	#display.EndRender()
	cv2.imshow('img', img)
	if cv2.waitKey(1)==ord(q):
		break
	
#	print out performance info
	display.SetTitle("Object Detection | Network {fps:.0f} FPS | Total_Vehicle_Counts = {counts}". format(fps=net.GetNetworkFPS(), counts=vehicleCount))
	
#	net.PrintProfilerTimes()

camera.release()
cv2.destroyAllWindows()

# load the object detection network
#net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
#camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
#display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
