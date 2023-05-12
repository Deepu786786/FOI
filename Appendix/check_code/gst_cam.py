# import jetson.inference
# import jetson.utils
# import cv2

# net = jetson.inference.detectNet(network="ssd-mobilenet-v2" ,threshold=0.1)

import jetson.utils
import argparse



parse the command line
parser = argparse.ArgumentParser()

parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0), or for VL42 cameras the /dev/video node to use (e.g. /dev/video0).  By default, MIPI CSI camera 0 will be used.")

opt = parser.parse_args()
print(opt)

# create display window
display = jetson.utils.glDisplay()

# create camera device
camera = jetson.utils.gstCamera(640,480,"rtspsrc location=rtsp://192.168.0.12:8080/video/h264 latency=0 ! queue ! rtph264depay ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink")

# open the camera for streaming
camera.Open()

# capture frames until user exits
while display.IsOpen():
	image, width, height = camera.CaptureRGBA()
	display.RenderOnce(image, width, height)
	display.SetTitle("{:s} | {:d}x{:d} | {:.0f} FPS".format("Camera Viewer", width, height, display.GetFPS()))
	
# close the camera
camera.Close()






























# camera_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=2 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=1920, height=1080 ! videoconvert ! xvimagesink"
# camera = jetson.utils.gstCamera(640,480,camera_url)

# # camera = jetson.utils.gstCamera(640,480,"rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=2 ! rtph264depay ! h264parse ! queue !  omxh264dec ! nvvidconv ! xvimagesink")
# # camera = jetson.utils.gstCamera("rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=2 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=1280, height=720 ! videoconvert ! xvimagesink")   
# # display = jetson.utils.glDisplay()

# while True:
# 	ret,frame=camera.read()
# 	cv2.imshow('mycamera',frame)

# 	if cv2.waitkey(1) & 0xFF == ord('q'):
# 		break


# while display.IsOpen():

# 	ret,frame = camera.read()
# 	if not ret:
# 		print("Error retrieving frame")
# 		break
	
# 	img = cv2.cvtColor(frame,cv2.COLOR_RGB2RGBA)

# 	img, width, height = camera.CaptureRGBA()
	
# 	detections = net.Detect(img, width, height, overlay='lines,labels,conf')
	
# 	vehicleCount=0
# 	for obj in detections:
# 		if obj.ClassID>1 and obj.ClassID<10:
# 			vehicleCount+=1



# #	display.RenderOnce(img, width, height)
# 	#display.BeginRender()
# 	#display.Render(img)
# 	#display.EndRender()
# 	cv2.imshow('img', img)
# 	if cv2.waitKey(1)==ord(q):
# 		break
	
# #	print out performance info
# 	display.SetTitle("Object Detection | Network {fps:.0f} FPS | Total_Vehicle_Counts = {counts}". format(fps=net.GetNetworkFPS(), counts=vehicleCount))
	
#	net.PrintProfilerTimes()

camera.release()
cv2.destroyAllWindows()


