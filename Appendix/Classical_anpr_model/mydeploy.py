'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
How to train custom yolov5: https://youtu.be/12UoOlsRwh8
DATASET: 1) https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation
         2) https://www.kaggle.com/datasets/elysian01/car-number-plate-detection
'''
### importing required libraries
import torch
import cv2
import time
# import pytesseract
import re
import numpy as np
#import easyocr


##### DEFINING GLOBAL VARIABLE
#EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
#OCR_TH = 0.2




### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates




# ### to filter out wrong detections 

# def filter_text(region, ocr_result, region_threshold):
#     rectangle_size = region.shape[0]*region.shape[1]
    
#     plate = [] 
#     print(ocr_result)
#     for result in ocr_result:
#         length = np.sum(np.subtract(result[0][1], result[0][0]))
#         height = np.sum(np.subtract(result[0][2], result[0][1]))
        
#         if length*height / rectangle_size > region_threshold:
#             plate.append(result[1])
#     return plate





### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None, vid_out = None):

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    #model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('./yolov5', 'custom', source ='local', path='last.pt') ### The repo is stored locally

    classes = model.names ### class names in string format



    ### --------------- for detection on image --------------------
    # if img_path != None:
    #     print(f"[INFO] Working with image: {img_path}")
    #     img_out_name = f"./output/result_{img_path.split('/')[-1]}"

    #     frame = cv2.imread(img_path) ### reading the image

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        # frame = plot_boxes(results, frame,classes = classes)
        

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f"[INFO] Exiting. . . ")

                cv2.imwrite(f"{img_out_name}",frame) ## if you want to save he output result.

                break
        


        ## closing all windows
        cv2.destroyAllWindows()



### -------------------  calling the main function-------------------------------


# main(vid_path="./test_images/vid_1.mp4",vid_out="vid_1.mp4") ### for custom video
main(vid_path=0,vid_out="webcam_facemask_result.mp4") #### for webcam

# main(img_path="./img1.jpeg") ## for image
            

