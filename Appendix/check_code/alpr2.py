import cv2
#import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread("images/img4.jpeg")
gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.imshow(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB))
#apply filter and edge detection
bfilter = cv2.bilateralFilter(gray,11,17,17)
edged = cv2.Canny(bfilter,30,200)
cv2.imshow("edged",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

#find contours and apply mask
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key =cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    apporx = cv2.approxPolyDP(contour,10,True)
    if len(apporx)==4:
        location =apporx
        break
print(location) 

mask =np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255,-1)
new_image = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("new_image",new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()   

(x,y) = np.where(mask==255)
(x1,y1) = (np.min(x),np.min(y))
(x2,y2) = (np.max(x), np.max(y))

cropped_image = gray[x1:x2+1,y1:y2+1]
cv2.imshow("cropped_image",cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#use easyocr to read text
reader =easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)

#render result
text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org =(apporx[0][0][0], apporx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA )
res = cv2.rectangle(img, tuple(apporx[0][0]),tuple(apporx[2][0]), (0,255,0), 3)
cv2.inshow("result",res)
cv2.waitKey(0)
cv2.destroyAllWindows()

