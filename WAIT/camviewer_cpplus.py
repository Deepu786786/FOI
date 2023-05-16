import cv2
from wait_functions import camset2, put_FPS

cam = camset2()

while True:
	ret, frame = cam.read()
	if ret:

		frame,fps = put_FPS(frame)
		
		cv2.imshow("Camera Feed", frame)
		
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	if not ret:
		print("Error retrieving frame.")

cam.release()
cv2.destroyAllWindows()
