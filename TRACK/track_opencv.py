import cv2
import jetson_inference
import jetson_utils
import numpy as np
from track_functions import camset2, detectVehicles,opencvToCuda, cudaToOpencv, put_Rect, put_Text, put_FPS


cam = camset2()
		

net = jetson_inference.detectNet(network="ssd-mobilenet-v2",threshold=0.5)

while True:
	ret, frame = cam.read()

	if ret:
		height,width = frame.shape[0],frame.shape[1]	
		cudaimg = opencvToCuda(frame)
		
		detections = detectVehicles(cudaimg,width,height,net)

		vehicleCount=0
		for detect in detections:
			ID = detect.ClassID
			item = net.GetClassDesc(ID)
			conf = detect.Confidence
			top, bottom, left, right = int(detect.Top), int(detect.Bottom), int(detect.Left), int(detect.Right)

			if ID>1 and ID<10:
				vehicleCount+=1

			frame = put_Rect(frame,top,left,bottom,right)
			frame = put_Text(frame, item, x=left-5, y=top, font_scale=0.5, color=(0,255,0), text_thickness=1)
			text = 'Confidence: ' + str(round(conf,4)*100) + '%'
			frame = put_Text(frame, text, x=left-5, y=top+15, font_scale=0.5, color=(0,255,0), text_thickness=1)

		frame = put_Text(frame, 'Vehicle Count = '+str(vehicleCount), x=200, y=10, font_scale=1, text_thickness=2)
		frame,fps = put_FPS(frame)


		cv2.imshow("Display", frame)


	if not ret:
		print("Error retrieving frame.")
		#exit()
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cam.release()
cv2.destroyAllWindows()





