import cv2
import jetson_inference
import jetson_utils
import numpy as np
from wait_functions import camset, detectPlate, characterSegmentation, reorientCudaimg, recognizePlate, getNumberPlate
from wait_functions import cudaToOpencv, put_Text, put_FPS

net_plate = jetson_inference.detectNet(model='./weights_plate/plate_ssdmobilenetv1.onnx', labels='./weights_plate/labels.txt', input_blob='input_0', output_cvg='scores', output_bbox='boxes', threshold=0.7)
# net_ocr = jetson_inference.detectNet(model='./weights_ocr/ocr_ssdmobilenetv1.onnx', labels='./weights_ocr/labels.txt', input_blob='input_0', output_cvg='scores', output_bbox='boxes', threshold=0.5)

camera = camset()


# display = jetson_utils.glDisplay()


# while display.IsOpen():
while True:

	cudaimg, width, height = camera.CaptureRGBA(zeroCopy=True)
	jetson_utils.cudaDeviceSynchronize()

	# img = reorientCudaimg(img,width,height,+15)

	plate_img,coords = detectPlate(cudaimg,width,height,net_plate)

	# plate_img = cv2.resize(plate_img,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC)
	# plate_img = cv2.resize(plate_img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

	plate_preprocessed = characterSegmentation(plate_img)

	# plate_preprocessed = cv2.resize(plate_preprocessed,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

	number = getNumberPlate(plate_preprocessed)

	# ocrimg,ocrwidth,ocrheight = recognizePlate(plate_preprocessed,net_ocr)
	# filtered_detections = apply_nms(bbox_list, confidence_threshold=0.7, iou_threshold=0.5)

	opcimg = cudaToOpencv(cudaimg,width,height)
	left = int(coords[0])
	bottom = int(coords[2])
	opcimg = put_Text(opcimg,number,left,bottom,2,(0,0,255),2)
	opcimg = put_FPS(opcimg)

	cv2.imshow('Number plate',opcimg)



	# display.SetTitle("WAIT SYSTEM | NetworkFPS = "+str(round(net_plate.GetNetworkFPS(),0))+" fps")

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.Close()
