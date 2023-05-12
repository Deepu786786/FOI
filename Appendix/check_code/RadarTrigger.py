import serial

ser = serial.Serial(
	port='/dev/ttyTHS1',
	baudrate=19200,
	timeout=5
)
while True:
	data=ser.readline().decode('utf-8',errors='ignore').strip()
	if data and data != "0.00":
		data = float(data)
		if data >= 5.00 or data <= -5.00:
		print(data)
