import jetson.inference
import jetson.utils
import serial
import time

#ser = serial.Serial(
#	port='/dev/ttyTHS1',
#	baudrate=19200,
#	timeout=10
#)


# def vehicleCount(detections):
# 	count=0
# 	for obj in detections:
# 		if obj.ClassID>1 and obj.ClassID<10:
# 			count+=1
# 	return count


# net = jetson.inference.detectNet(network="ssd-mobilenet-v2" ,threshold=0.5)

# camera = jetson.utils.gstCamera(1280, 720,"rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=2 ! rtph264depay ! h264parse ! queue !  omxh264dec ! nvvidconv ! xvimagesink")
camera = jetson.utils.gstCamera(1280, 720,"rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=2 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=1280, height=720 ! videoconvert ! xvimagesink")    
# camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()

timestamp = time.time()
fpsfilt=0
while display.IsOpen():
	img, width, height = camera.CaptureRGBA()
	detections = net.Detect(img, width, height, overlay='lines,labels,conf')
	
	
	#data=ser.readline().decode('utf-8',errors='ignore').strip()
	#if data >= "5.00" or data <= "-5.00":
	#	print(data)
	#else:
	#	pass




	display.RenderOnce(img, width, height)
	#display.BeginRender()
	#display.Render(img)
	#display.EndRender()
	
	dt=time.time()-timestamp
	fps=1/dt
	fpsfilt=0.9*fpsfilt+0.1*fps
	timestamp=time.time()
	
	print('algorihtm fps: ',str(round(fpsfilt,2))	)
	print('fps: ',net.GetNetworkFPS())

	display.SetTitle("Object Detection | Network {fps:.0f} FPS | Total_Vehicle_Counts = {counts}". format(fps=net.GetNetworkFPS(), counts=vehicleCount(detections)))
	
#	net.PrintProfilerTimes()
