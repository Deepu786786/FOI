from wait_functions import camset2
import cv2
cam = camset2()

while True:


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()