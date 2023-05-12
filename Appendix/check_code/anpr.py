import cv2
import pytesseract
import numpy as np
import imutils
from skimage.segmentation import clear_border


# img = cv2.imread(pwd+"Numberplate_images/goodplates/numberplate_5.jpeg")
img = cv2.imread("Numberplate_images/numberplate_3.jpeg")	
cv2.imshow("image",img)
cv2.waitkey(0)

#*** Enlarging ***
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)

#*** Sharpening ***
gaussian_blur = cv2.GaussianBlur(gray,(7,7),5)
sharpen = cv2.addWeighted(gray,3.5,gaussian_blur,-2.5,2)

#*** Blackhat morphological transformation ***
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
blackhat = cv2.morphologyEx(sharpen, cv2.MORPH_BLACKHAT, rectKern)


#Shows the region that includes the license plate standing out
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
light = cv2.morphologyEx(gray,cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


#*** Edge emphasizing ***
scharr = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
scharr = np.absolute(scharr)
(minVal, maxVal) = (np.min(scharr), np.max(scharr))
scharr = 255 * ((scharr - minVal) / (maxVal - minVal))
scharr = scharr.astype("uint8")

#   from packaging import version
#ModuleNotFoundError: No module named 'packaging'canny = cv2.Canny(blackhat,400,700)


scharr = cv2.GaussianBlur(scharr, (5,5),0)
scharr = cv2.morphologyEx(scharr, cv2.MORPH_CLOSE, rectKern)
thresh = cv2.threshold(scharr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#canny = cv2.GaussianBlur(canny, (5,5),0)
#canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, rectKern)
#thresh2 = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


thresh = cv2.erode(thresh,None, iterations=3)
thresh = cv2.dilate(thresh,None, iterations=3)


thresh2 = cv2.bitwise_and(thresh, thresh, mask=light)
thresh2 = cv2.dilate(thresh2, None, iterations=4)
thresh2 = cv2.erode(thresh2, None, iterations=4)


#*** Finding contours ***
cnts,hierarchy = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

#sharpen_copy = sharpen.copy()
#sharpen_copy = cv2.drawContours(sharpen_copy, cnts, -1, (0,0,0), thickness=3)
#cv2.imshow('Drawn Contours', sharpen_copy)

#Filtering from selected top 5 contours

#sharpen_copy2 = sharpen.copy()
#color = (0,0,0)
#thickness=3

filtered_contours=[]
for c in cnts:
	x,y,w,h = cv2.boundingRect(c)

	#filtering using contour area	
	platearea = cv2.contourArea(c) 
	if platearea <=1500 or platearea >=5000:
		continue
	
	#filtering using extent
	rect_area = w*h
	extent = float(platearea)/rect_area
	if extent <= 0.3:
		continue
	
	#filtering using aspect ratio
	aspect_ratio = w/float(h)
	if aspect_ratio<=1.6 or aspect_ratio>=7:
		continue

	#filtering using rectangularity
	_,_, angle = cv2.minAreaRect(c)
	if angle < -45:
		angle += 90
	
	rectangularity = platearea/rect_area
	if rectangularity <= 0.35:
		continue

#	print(w/float(h))
	#cv2.rectangle(sharpen_copy2, (x,y),(x+w,y+h), color, thickness)
	#cv2.putText(sharpen_copy2, str(rectangularity), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
	
	filtered_contours.append(c)

filtered_contours.sort(key=cv2.contourArea, reverse=True)
license_plate_contour = filtered_contours[0]

if license_plate_contour is None:
	print("License plate is not detected")
	cv2.destroyAllWindows()


x,y,w,h = cv2.boundingRect(license_plate_contour)
x=x-10
w=w+15
y=y-10
h=h+20

# a,b = (int(0.02*sharpen.shape[0]), int(0.025*sharpen.shape[1]))
# licensePlate = sharpen[y+a:y+h-a, x+b:x+w-b]

licensePlate = sharpen[y:y+h, x:x+w]

licensePlate = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
licensePlate2 = clear_border(licensePlate)
cv2.imshow('licensePlate', licensePlate)
cv2.imshow('licensePlate2', licensePlate2)
cv2.waitKey(0)



#********************************************** CHARACTER SEGMENTATION *****************************************************************








# #***************************************************** O C R ***************************************************************************

# #Whitelisting: telling Tesseract to only OCR alphanumeric characters
# alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# options = "-c tessedit_char_whitelist={}".format(alphanumeric)

# #setting the PSM mode (Page Segmentation Method)
# #Tesseract's setting has 13 modes of operation, but we will default to 7 -"treat the image as single text line"
# psm=7
# options += "--psm {}".format(psm)

# #licensePlate = clear_border(licensePlate)

# platenumber = pytesseract.image_to_string(licensePlate,config=options)
# print(platenumber)

# img_copy = img.copy()
# img_copy=cv2.resize(img_copy,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
# cv2.rectangle(img_copy, (x,y),(x+w,y+h), (0,255,0), 5)
# img_copy = cv2.resize(img_copy,None,fx=1/2.5,fy=1/2.5,interpolation=cv2.INTER_AREA)
# cv2.imshow('original',img_copy)

# #cv2.imshow('gray', gray)
# #cv2.imshow('sharpen',sharpen)
# #cv2.imshow('blackhat', blackhat)
# #cv2.imshow('light', light)
# #cv2.imshow('Scharr', scharr)
# #cv2.imshow('thresh',thresh)
# #cv2.imshow('thresh2',thresh2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
