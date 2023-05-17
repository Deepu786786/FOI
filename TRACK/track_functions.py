import cv2
import jetson_utils
import numpy as np
import time


def camset():
    
    # camera = jetson_utils.gstCamera(640, 480, camera_url)
    camera = jetson_utils.gstCamera(640, 480, '/dev/video0')

    return camera



def camset2():
	# gst-launch-1.0 rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=0 ! queue ! rtph264depay ! queue ! h264parse ! queue !  omxh264dec ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink

    # camera_url = "rtsp://admin:Dd22864549*@10.13.1.60:554/cam/realmonitor?channel=1&subtype=0"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=480, height=360 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=640, height=480 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=1280, height=720 ! appsink drop=1"
    camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=1920, height=1080 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! appsink"
   
    # cam = cv2.VideoCapture(camera_url, cv2.CAP_GSTREAMER)
    
    cam = cv2.VideoCapture('/dev/video0')
    # cam.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    if not cam.isOpened():
        print("Error opening RTSP stream.")

    return cam



def put_Text(frame,text='NoText', x=10, y=10, font_scale=2, color=(0,0,255), text_thickness=1):

	if isinstance(text,float) or isinstance(text,int):
		text = str(round(text,2))
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	text_size = cv2.getTextSize(text,font, font_scale, text_thickness)[0]
	text_x = x + 10
	text_y = y + 15

	return cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, text_thickness)



timestamp = time.time()
fpsfilt=0
def put_FPS(frame):
	global timestamp, fpsfilt
	dt = time.time()-timestamp
	timestamp=time.time()
	fps=1/dt
	fpsfilt = 0.9*fpsfilt+0.1*fps
	
	text = 'FPS: '+str(round(fpsfilt,2))
	frame = put_Text(frame,text,x=5,y=10,font_scale=1,text_thickness=2)

	return frame,text



def put_Rect(img,top,left,bottom,right):
	green_color = (0,255,0)
	thickness = 1
	start_point = (left,top)
	end_point = (right,bottom)

	img = cv2.rectangle(img, start_point, end_point, green_color, thickness)

	return img



def opencvToCuda(frame):
    cudaimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    cudaimg = jetson_utils.cudaFromNumpy(cudaimg)
    return cudaimg



def cudaToOpencv(cudaimg,width,height):

    numpy_array = jetson_utils.cudaToNumpy(cudaimg,width,height,4)
    frame = cv2.cvtColor(numpy_array.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    return frame



def reorientCudaimg(cudaimg,width,height,angle):

    frame = cudaToOpencv(cudaimg,width,height)

    h,w,c =  frame.shape
    center = (h/2,w/2)

    rotation_matrix = cv2.getRotationMatrix2D(center,angle,1.0)
    rotatedframe = cv2.warpAffine(frame,rotation_matrix,(w,h))

    return opencvToCuda(rotatedframe)



def detectVehicles(cudaimg,width,height,net):
    return net.Detect(cudaimg,width,height)



# #Non-Maximum Suppression to discard redundant and overlapping bounding boxes
# def apply_nms(detections, confidence_threshold=0.7, iou_threshold=0.5):
#     # Sort detections based on confidence scores (descending order)
#     detections = sorted(detections, key=lambda x: x[4], reverse=True)

#     # List to store the filtered detections
#     filtered_detections = []

#     # Loop through each detection
#     while len(detections) > 0:
#         # Get the highest confidence detection
#         best_detection = detections[0]
#         filtered_detections.append(best_detection)

#         # Compute the Intersection over Union (IoU) with other detections
#         ious = []
#         for detection in detections[1:]:
#             iou = calculate_iou(best_detection, detection)
#             ious.append(iou)

#         # Discard detections with IoU greater than the threshold
#         detections = [d for d, iou in zip(detections[1:], ious) if iou < iou_threshold]

#     # Filter detections based on confidence threshold
#     filtered_detections = [d for d in filtered_detections if d[4] >= confidence_threshold]

#     return filtered_detections


# def calculate_iou(box1, box2):
#     # Calculate coordinates of the intersection rectangle
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     # Calculate the area of intersection rectangle
#     intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

#     # Calculate the areas of both bounding boxes
#     box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
#     box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

#     # Calculate the Intersection over Union (IoU)
#     iou = intersection_area / float(box1_area + box2_area - intersection_area)

#     return iou
