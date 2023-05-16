import cv2
import jetson_inference
import jetson_utils
import numpy as np
from wait_functions import camset, detectPlate, characterSegmentation, reorientCudaimg, recognizePlate, getNumberPlate
from wait_functions import cudaToOpencv, put_Text, put_FPS

# net_plate = jetson_inference.detectNet(model='./weights_plate/plate_ssdmobilenetv1.onnx', labels='./weights_plate/labels.txt', input_blob='input_0', output_cvg='scores', output_bbox='boxes', threshold=0.7)

camera = camset()
# display = jetson_utils.glDisplay()


# while display.IsOpen():
while True:

	cudaimg, width, height = camera.CaptureRGBA(zeroCopy=True,format='rgba32f')
	jetson_utils.cudaDeviceSynchronize()
	# img = reorientCudaimg(img,width,height,+15)

	frame = cudaToOpencv(cudaimg,width,height)

	# plate,coords = detectPlate(cudaimg,width,height,net_plate)
	# # plate = cv2.resize(plate,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC)
	# # plate = cv2.resize(plate,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

	# plate_preprocessed = characterSegmentation(plate)
	# number = getNumberPlate(plate_preprocessed)

	
	# # left = int(coords[0])
	# # bottom = int(coords[2])
	# # frame = put_Text(frame,number,left,bottom,2,(0,0,255),2)
	frame,fps = put_FPS(plate)

	cv2.imshow('Number plate',frame)

	# print(number,fps)


	# display.SetTitle("WAIT SYSTEM | NetworkFPS = "+str(round(net_plate.GetNetworkFPS(),0))+" fps")
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.Close()
