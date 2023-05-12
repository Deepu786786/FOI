import jetson.inference
import jetson.utils


camera =jetson.utils.gstCamera(1280,720, "rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554 latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! videoconvert ! xvimagesink")
# camera = jetson.utils.gstCamera(1920,1080,"rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=0 ! rtph264depay ! h264parse ! queue ! omxh264dec ! nvvidconv ! video/x-raw,format='BGRx' ! videoconvert ! video/x-raw, foramt='BGR' ! xvimagesink")
# camera =jetson.utils.gstCamera(1280,720, "/dev/video0")
# "rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554 latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! xvimagesink"

print("hello")    

# camera = jetson.utils.gstCamera(640, 480, "filesrc location=/opt/nvidia/deepstream/deepstream-4.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ")
# import jetson.utils
# import argparse

# camera = jetson.utils.gstCamera(640, 480, "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=0 ! rtph264depay !  h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! xvimagesink")

# # parse the command line
# #parser = argparse.ArgumentParser()

# #parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
# #parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
# #parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0), or for VL42 cameras the /dev/video node to use (e.g. /dev/video0).  By default, MIPI CSI camera 0 will be used.")

# #opt = parser.parse_args()
# #print(opt)

# # create display window
# display = jetson.utils.glDisplay()

# # create camera device
# camera = jetson.utils.gstCamera(640,480,"rtspsrc location=rtsp://192.168.0.12:8080/video/h264 latency=0 ! queue ! rtph264depay ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink")

# # open the camera for streaming
# camera.Open()

# # capture frames until user exits
# while display.IsOpen():
# 	image, width, height = camera.CaptureRGBA()
# 	display.RenderOnce(image, width, height)
# 	display.SetTitle("{:s} | {:d}x{:d} | {:.0f} FPS".format("Camera Viewer", width, height, display.GetFPS()))
	
# # close the camera
# camera.Close()



# import jetson.inference
# import jetson.utils
# import cv2

# # net = jetson.inference.detectNet(network="ssd-mobilenet-v2" ,threshold=0.1)


# # camera_url = "rtsp://admin:Dd22864549*@10.13.1.60:554/cam/realmonitor?channel=1&subtype=0"
# # camera = cv2.VideoCapture(camera_url)

# # camera = jetson.utils.gstCamera(640,480,"rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=2 ! rtph264depay ! h264parse ! queue !  omxh264dec ! nvvidconv ! xvimagesink")
# camera = jetson.utils.gstCamera("rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=2 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=1280, height=720 ! videoconvert ! xvimagesink")   
# display = jetson.utils.glDisplay()