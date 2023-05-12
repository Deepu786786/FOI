import cv2
import jetson_utils
import time


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


def camSet():
	camera_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via switch
	#camera_url = "rtsp://admin:Dd22864549*@192.168.100.159:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 4MP camera when connected via 4G router
	#camera_url = "rtsp://admin:Dd22864549*@192.168.100.160:554/cam/realmonitor?channel=1&subtype=0" #Cpplus 8MP camera when connected via 4G router


	# # gstCamera: It is C++ library that provides a low-level API for capturing video frames using the GStreamer pipeline on Jetson platforms.
	# # It provides finer control over the GStreamer pipline and allows you to customizrtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsinke the pipeline to suit your specific needs
	gstremer_cam_pipeline = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"
	# disW = 640
	# disH = 480
	# disW = 1280
	# disH = 960
	# cam = jetson_utils.gstCamera(1280,720,camera_url)


	import gi
	gi.require_version('Gst','1,0')
	from gi.repository import Gst

	Gst.init(None)
	pipeline = Gst.parse_launch(gstremer_cam_pipeline)
	pipeline.set_state(Gst.State.PLAYING)

	loop = GObject.MainLoop()
	loop.run()



	# cam = cv2.VideoCapture("rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=1 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER)
	# cam = cv2.VideoCapture(0)
	
	# cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

	# if not cam.isOpened():
	# 	print("Error opening RTSP stream.")
	# 	exit()
	
	return cam



cam = camSet()

while True:
	ret, frame = cam.read()
	if ret:
# 		# frame = cv2.resize(frame,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
# 		# frame = cv2.resize(frame,(640,480),interpolation=cv2.INTER_AREA)

		frame = put_FPS(frame)
		cv2.imshow("Camera Feed", frame)
		
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	if not ret:
		print("Error retrieving frame.")

cam.release()
cv2.destroyAllWindows()



# display = jetson_utils.glDisplay()

# while not display.IsClosed():
# 	img,width,height = cam.CaptureRGBA(zeroCopy=1)
	
# 	display.RenderOnce(img,width,height)

# cam.Close()


#8MP camera mainstream resolution: 3840*2160
