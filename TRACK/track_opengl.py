import jetson_inference
import jetson_utils
# from track_functions import camset


camera = camera = jetson_utils.gstCamera(640, 480, '/dev/video0')

net = jetson_inference.detectNet(network="ssd-mobilenet-v2",threshold=0.4)

display = jetson_utils.glDisplay()

while display.IsOpen():

	cudaimg, width, height = camera.CaptureRGBA()
	
	detections = net.Detect(cudaimg,width,height)
	
	vehicleCount=0
	for detect in detections:
		ID = detect.ClassID
		item = net.GetClassDesc(ID)
		left = int(detect.Left)
		top = int(detect.Top)
		bottom = int(detect.Bottom)
		right = int(detect.Right)

		if ID>1 and ID<10:
			vehicleCount+=1
	

	display.RenderOnce(cudaimg, width, height)
	
	display.SetTitle("TRACK SYSTEM | Network {fps:.0f} FPS | Vehicle_Counts = {counts}". format(fps=net.GetNetworkFPS(), counts=vehicleCount))
	
#	net.PrintProfilerTimes()

camera.Close()
