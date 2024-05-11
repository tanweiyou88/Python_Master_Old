from ultralytics import YOLO
import cv2
import os
import csv

results = [] # define empty dictionary

# Load a model
# model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights

# # Train the model
# results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category
# Predict with the model

# Open the video file
video_path = 'D:\Pose_estimation_YOLOv8_AarohiSingla\Juru _ch1_20240115162905_20240115163008_COPY_short.mp4'
cap = cv2.VideoCapture(video_path)
a =-1
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    # a += ret
    if ret:
        # Run the YOLOv8 inference on the frame
        results = model(frame, imgsz=320, save=True, show=False)[0]
        for result in results:
            for keypoint in result.keypoints.tolist():
                print(keypoint)
        # for result in results:
        #     for keypoint in result.keypoints.tolist():
        #         if keypoint is not None: 
        #             print(keypoint) 
        #             keypoint_results = {'Keypoint Coordinate':{[keypoint]}}
                results.append(keypoint)

        # Visualize the results on the frame
        annotated_frame=results.plot()

        # # Display the annotated frame
        cv2.imshow("YOLOv8 Inference",annotated_frame)

        # Break the loop if 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached 
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# ## write results
# with open('D:\Pose_estimation_YOLOv8_AarohiSingla\Keypoint_results.csv','w',newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(results)





