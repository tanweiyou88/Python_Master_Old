"""
Project: Automatic License Plate Recognition (ALPR), based on YOLOv8, SORT & EasyOCR
Aim: Serves as the complete pipeline to detect vehicles, to detect and recognize license plates, and to store the detection information of the detected vehicles and license plates on a csv file.
ALPR Youtube tutorial repository: https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8
Simple Online and Realtime Tracking (SORT) algorithm repository: https://github.com/abewley/sort

Complete step of the ALPR project:
1) Update the functions in "util.py" to set/fix the license plate format in the project
2) Execute "ALPR_Phillip_main.py" to perform detections & recognitions with the functions in "util.py". At the end, get a csv file that stores the results.
3) Update the location & name of the original & interpolated csv files respectively in "add_missing_data.py", then execute "add_missing_data.py" to get the interpolated csv file.
4) In "visualize.py", update the location & name of the interpolated csv file, the location & name of the video used to perform detections & recognitions in "ALPR_Phillip_main.py", 
and the location & name of the video which added with the bounding boxes/markings (visualize the information stored in the interpolated csv file). Then, execute "visualize.py" to 
get the video which added with the bounding boxes/markings.
"""

import os
from ultralytics import YOLO
import cv2

from sort.sort import * # from the library called "sort", import everything
from util import get_car, read_license_plate, write_csv # from the "util.py" file, import the defined function "get_car"

results = {} # define empty dictionary
mot_tracker = Sort() # Sort() is the object tracker. It will track an object based on the given bounding box information. (It will be used to track vehicles in this project)

## load models
coco_model = YOLO('yolov8n.pt') # load "yolov8n.pt" as the object detector for car (car detector) based on YOLOv8 nano. We named this car detector as "coco_model". "yolov8n.pt" or YOLOv8 nano is a pre-trained model (pre-trained using COCO dataset with 80 pre-trained class, according to Github).
license_plate_detector=YOLO(os.path.join('D:\Python_Master', 'runs', 'detect', 'train16_Model_LicensePlateDetector_Dataset_Car_and_License_Plate_from_jonG312_GitHub_400Epochs', 'weights', 'last.pt')) # load my trained machine learning model (through its location & name) as license plate detector

## load video
cap = cv2.VideoCapture(os.path.join('D:\Python_Master\Automatic_License_Plate_Recognition_by_Phillip', 'Data', 'Highway_Traffic_Flow.mp4')) # load the video (through its location & name) which used for detections & recognitions
vehicles = [2, 3, 5, 7] # The class ID of car, motorbike, bus, and truck, according to COCO dataset

## read frames
frame_number=-1
ret = True
while ret:
        frame_number+=1 # to increment the threshold -> frame number
        ret, frame =cap.read() # read the frames of the video, one frame at a time
        if ret: # if there is a video frame (the video not yet ended)
            # if frame_number > 10: # "frame_number > 10" specifies the condition we only execute the first 10 frames of the video (For the sake of testing this pipeline file). After executing the first 10 frames, it will exit the while loop. Can comment this if block to perform detections & recognitions over the whole video
            #      break
            results[frame_number]={} # the "frame_number" is the key of this dictionary. The results at different frames will be saved to the dictionary at different keys.
            ## detect all vehicles on the frame
            detections = coco_model(frame)[0] # input the frame of the video to the machine learning model to perform object detection (The model can detect up to 80 objects because it is pretrained on the COCO dataset). [0] means one step at a time. At this code, multiple detections (objects detected) are obtained on this(single) frame. 
            detections_=[] # create an empty variable
            for detection in detections.boxes.data.tolist(): # iterate all the detections we obtained from the machine learning model on this frame
                #  print(detection) # print the detection information of all detections obtained at the terminal. For each detection on this frame, its detection information has 6 values such that [x1,y1,x2,y2,score,class_id]
                x1,y1,x2,y2,score,class_id = detection # For each detection on this frame, save 6 values of its detection information into 6 different variables respectively.
                if int(class_id) in vehicles: #  since the model is pretrained on COCO dataset, it can detect many different objects. This "if" statement specifies the condition that only the detection information of the detections (object detecteds) on this frame whose own class ID equals to either value stored in "vehicle", will be processed with the codes/ statements in the section/block below
                    detections_.append([x1,y1,x2,y2,score]) # Append/save the information of bounding box and confidence score of the detection to this variable called "detections_"       
            
            ## track all vehicles on the frame
            # tracking vehicles is important because by knowing the vehicle ID (every vehicle has an unique ID) of the vehicle we are interested, we can trace and determine its license plate on the csv file storing the results (When a vehicle ID is detected and recognized to have different license plate, we can determine the correct license plate by choosing the license plate with the highest condifence score on the csv file) 
            track_ids = mot_tracker.update(np.asarray(detections_)) # "track_ids" contains the bounding box of all the vehicles we detected on this frame. It will add a vehicle ID (an unique number for each vehicle detected) as an additional column/field. Each vehicle ID will represent a specific vehicle thorughout the video (so that each vehicle detected can be treated as a different object under the same class). "track_ids" does not store the class ID, because this information is not important from this step onwards.
            
            ## detect all license plates on the frame
            license_plates = license_plate_detector(frame)[0] # At this code, multiple detections (license plates detected) are obtained on this(single) frame.
            for license_plate in license_plates.boxes.data.tolist():
                x1,y1,x2,y2,score,class_id = license_plate # For each detection on this frame, save 6 values of its detection information into 6 different variables respectively. 
                
                ## assign each license plate to a vehicle on the frame (to ensure which license plate belongs to which vehicle)
                xcar1,ycar1,xcar2,ycar2,vehicle_id = get_car(license_plate, track_ids) # "get_car" function returns the coordinates of the car each license plate detected on this frame belongs to
                
                if vehicle_id != -1: # for the case the license plate is successfully assigned to a specific vehicle
                    ## crop each license plate on the frame
                    license_plate_crop = frame[int(y1):int(y2),int(x1):int(x2), :] # input the coordinates of each license plate detected on this frame
                    
                    ## process each license plate on the frame (apply some image processing filters to each cropped license plate to improve the image so it is much simpler for the OCR technology to read the content from the image at later stage) [To convert each license plate from BGR into binary]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) # convert the cropped license plate from BGR (3 dimensions) to grayscale (1 dimension)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV) # convert the grayscale cropped license plate into binary (black & white). The "license_plate_crop_thresh" stores the binary cropped license plate and it will be the input for the OCR technology later. "_" stores the information which are not useful in this project.
                    
                    ## read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh) # this function will return each formatted license plate text and its confidence score

                    if license_plate_text is not None: # when a license plate is succesfully read
                        # the results dictionary only stores the detection information of:
                        # 1) detected vehicles which are successfully assigned with a license plate
                        # 2) detected license plates whose format are complied to the desired format of this project  
                        results[frame_number][vehicle_id] = {'car':{'bbox': [xcar1, ycar1, xcar2, ycar2]}, # the structure of this dictionary has 2 keys: frame number & car ID. Because the information stored in the dictorary is tied to a specific car on a specific frame.
                                                        'license_plate':{'bbox':[x1,y1,x2,y2],'text':license_plate_text,'bbox_score':score,'text_score':license_plate_text_score}}

## write results
write_csv(results,os.path.join('D:\Python_Master\Automatic_License_Plate_Recognition_by_Phillip','ALPR_results.csv')) # Parameters: the dictionary whose information will be save into a csv file, the path (location & name) we want to save this csv file. The best practice to define the path for a file is to use the "os.path.join()", which ensures compatibility across different operating systems.
   

