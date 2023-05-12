import cv2


# a location for the camera stream
camera_stream = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"

cap =cv2.VideoCapture(camera_stream,cv2.CAP_GSTREAMER)

cuda_decoder = cv2.cudacodec.createVideoReader("h264")

while True:
    ret,frame = cap.read()
    if not ret:
        print("ERROR Retieving Frame")
        break
    cuda_frame = cuda_decoder.decode(frame)

    cpu_frame = cuda_frame.download()
    cv2.imshow("ip_camera",cpu_frame)

    if cv2.waitkey(1)& OxFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    

