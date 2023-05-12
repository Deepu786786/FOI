import jetson.inference
import jetson.utils

# net = jetson.inference.detectNet(model='./weights_plate/plate_ssdmobilenetv1.onnx', labels='./weights_plate/labels.txt', input_blob='input_0', output_cvg='scores', output_bbox='boxes', threshold=0.2)
net = jetson.inference.detectNet(model='./weights_ocr/ocr_ssdmobilenetv1.onnx', labels='./weights_ocr/labels.txt', input_blob='input_0', output_cvg='scores', output_bbox='boxes', threshold=0.1)

camera_url = "rtsp://admin:Dd22864549*@10.13.1.60:554/cam/realmonitor?channel=1&subtype=0"

# camera = jetson.utils.gstCamera(640, 480, "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0 latency=0 ! rtph264depay ! h264parse ! nvv412decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480")
# camera = jetson.utils.gstCamera(640, 480, camera_url)
camera = jetson.utils.gstCamera(1280,720,'/dev/video0')

display = jetson.utils.glDisplay()
display.SetTitle("WAIT SYSTEM")


while display.IsOpen():

	img, width, height = camera.CaptureRGBA(zeroCopy=True)
	
	detections = net.Detect(img, width, height, overlay='lines,labels,conf')
	
	vehicleCount=0
	carcount=0
	for detect in detections:
		ID = detect.ClassID
		item = net.GetClassDesc(ID)
		left=detect.Left
		top=detect.Top
		bottom=detect.Bottom
		right=detect.Right
		print(item)

		# if ID>1 and ID<10:
		# 	vehicleCount+=1
		# if item=='car':
		# 	carcount+=1
	

	display.RenderOnce(img, width, height)
	#display.BeginRender()
	#display.Render(img)
	#display.EndRender()
	
#	print out performance info
	display.SetTitle("Object Detection | Network {fps:.0f} FPS | Vehicle_Counts = {counts}". format(fps=net.GetNetworkFPS(), counts=vehicleCount))
	
#	net.PrintProfilerTimes()

display.Close()
camera.release()
