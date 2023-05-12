import serial
import pytesseract
import cv2

ser = serial.Serial(port='/dev/ttyTHS1', baudrate=19200, timeout=5)

camera_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
	print("Error opening RTSP stream.")
	exit()

def plateRecognizer(frame):
	pass


while display.IsOpen():
	data=ser.readline().decode('utf-8',errors='ignore').strip()
	if data and data != "0.00":
		data = float(data)
		if data >= 5.00 or data <= -5.00:	
			print(data)

			ret, frame = cap.read()

			if ret:
				cv2.imshow("Camera Feed", frame)
			if not ret:
				print("Error retrieving frame.")
			
			plateRecongnizer(frame)
			
			break

cap.release()
cv2.destroyAllWindows()

