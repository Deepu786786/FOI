#import jetson.inference
import jetson.utils
import cv2

#net = jetson.inference.detectNet(network="ssd-mobilenet-v2" ,threshold=0.1)


##Using OpenGL Library
#disW = 960
#disH = 720
#camera = jetson.utils.gstCamera(disW, disH, "/dev/video0")
#display = jetson.utils.glDisplay()
#while display.IsOpen():
#	img, width, height = camera.CaptureRGBA()

##	detections = net.Detect(img, width, height, overlay='lines,labels,conf')
##	vc=totalVehicles(detections)
##	net.PrintProfilerTimes()
##	display.SetTitle("Object Detection | Network {fps:.0f} FPS | Total_Vehicle_Counts = {counts}". format(fps=net.GetNetworkFPS(), counts=vc))

#	display.BeginRender()
#	display.Render(img)
#	display.EndRender()
	


##Using videoSource & videoOutput Library
#camera = jetson.utils.videoSource("v4l2:///dev/video0")
#while True:
#	frame = camera.Capture()
#	cv2.namedWindow("LogitechCam", cv2.WINDOW_NORMAL)
#	cv2.imshow("LogitechCam",frame)
#	if cv2.waiKey(1)==ord('q'):
#		break

#camera.Close()
#cv2.destroyAllWindows()



#Using OpenCV Library
winH=1280
winW=960
camera = cv2.VideoCapture("/dev/video0")     
#while True:
#	ret, frame = camera.read()
frame = cv2.imread("russian_number_plate.jpeg")	
frame = cv2.resize(frame,(winH,winW))

#preprocess the frame
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,3)
gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#	Detect the number plate
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
if plate_cascade.empty():
	print("Error: Cascade Classifier file not found or cannot be loaded.")
	exit()

plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
if len(plates) == 0:
	print("Error: Number plate not detected.")
	exit()
print (len(plates))    
#	Extract the number plate
#	for (x,y,w,h) in plates:
#		plate_img = frame[y:y+h, x:x+w]
#		cv2.imshow("LogitechCam2",plate_img)
#		cv2.moveWindow("LogitechCam2",0,0)

#	Recongnize the characters on the number plate

#	cv2.imshow("LogitechCam",gray)
#	cv2.moveWindow("LogitechCam",0,0)
	
#	if cv2.waitKey(1)==ord('q'):
#		break

#camera.release()
#cv2.destroyAllWindows()
