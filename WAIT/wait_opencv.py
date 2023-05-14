import cv2
import jetson_inference
import jetson_utils
import time
import numpy as np
from wait_functions import camset2, detectPlate, put_Rect, put_FPS, put_Text, characterSegmentation, getNumberPlate

cam = camset2()

net = jetson_inference.detectNet(model='./weights_plate/plate_ssdmobilenetv1.onnx.1.1.8201.GPU.FP16.engine', labels='./weights_plate/labels.txt', input_blob='input_0', output_cvg='scores', output_bbox='boxes', threshold=0.5)

	
while True:
	ret, frame = cam.read()

	if ret:
		height,width = frame.shape[0],frame.shape[1]	
		cudaimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
		cudaimg = jetson_utils.cudaFromNumpy(cudaimg)

		plate,coords = detectPlate(cudaimg,width,height,net)
		left = int(coords[0])
		top = int(coords[1])
		bottom = int(coords[2])
		right = int(coords[3])

		frame = put_Rect(frame,top,left,bottom,right)
		frame = put_FPS(frame)

		# plate = cv2.resize(plate,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
		# preprocessed_plate = characterSegmentation(plate)
		# number = getNumberPlate(preprocessed_plate)

		# frame = put_Text(frame,number,left,bottom,2,(0,0,255),2)

		# cv2.imshow("Number plate", plate)
		cv2.imshow("frame", frame)


	if not ret:
		print("Error retrieving frame.")
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cam.release()
cv2.destroyAllWindows()

#To increase kernel socket buffer max size on receiver side(jetson)
#sudo sysctl -w net.core.rmem_max=26214400





