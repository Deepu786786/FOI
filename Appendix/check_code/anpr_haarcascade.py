import cv2
import pytesseract
import os
import numpy as np


pwd = os.getcwd()
#frame = cv2.imread("jetson-inference/data/images/N58.jpeg")
frame = cv2.imread("Numberplate_images/numberplate_0.jpeg")	


gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


#Detecting and localize all number plates from the frame

plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
if plate_cascade.empty():
	print("Error: Cascade Classifier file not found or cannot be loaded.")
	exit()

plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
if len(plates) == 0:
	print("Error: Number plate not detected.")
	exit()

for (x,y,w,h) in plates:
	a,b = (int(0.02*frame.shape[0]), int(0.025*frame.shape[1]))
	plate = frame[y+a:y+h-a, x+b:x+w-b,:]
	kernel = np.ones((1,1), np.uint8)
	plate = cv2.dilate(plate, kernel, iterations=1)
	plate = cv2.erode(plate,kernel,iterations=1)
	#cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
	
	cv2.imshow('plate',plate)

#Character Segmentation



#Optical Character Recongnition (OCR) to recognize the extracted characters

text = pytesseract.image_to_string(plate)
print("Number plate: ", text)

while True:
	cv2.imshow("Detected Plate", frame)
	cv2.imshow("Plate_image", plate_img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#Challenges:
#1)Large and robust ANPR datasets for training/testing are difficult to obtain because ANPR companies and government entities closely guarding these datasets as proprietary information
