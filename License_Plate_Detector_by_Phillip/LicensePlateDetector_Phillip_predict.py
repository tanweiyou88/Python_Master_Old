# Apply the trained machine learning model/object detector on a video to predict the objects appear on the video
import os

from ultralytics import YOLO
import cv2

## read video
# Concatenate Base directory + folder + file name = image_path='D:\Python_Master\Learn_OpenCV_InputOutputVideo\data\Despicable_Me_2.mp4'
video_path = os.path.join('D:\Python_Master\License_Plate_Detector_by_Phillip\Data','Car_and_License_Plate_from_jonG312_GitHub','car_-_2165 (540p).mp4')

video_path_out = '{}_out.mp4'.format(video_path) # Save the same video but with prediction made by the machine learning model selected, with specified filename at the specified path.

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('D:\Python_Master', 'runs', 'detect', 'train15_Car_and_License_Plate_from_jonG312_GitHubCar_and_License_Plate_from_jonG312_GitHub_200Epochs', 'weights', 'last.pt') # the path where the machine learning model we want to select (usually our trained model) is located. "last.pt" is our trained machine learning model.

# Load a model
model = YOLO(model_path)  # load a custom model (Select a machine learning model)

threshold = 0.5

while ret:

    results = model(frame)[0] # model refers to the machine learning model we selected."(frame)" means we apply our model on each frame of the video, 1 frame at a time. "0" means we start to access the video from its first element/frame.

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold: # when an object on a video frame is detected by the machine learning model with probability (confidence score) > threshold
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4) # a shape (EG: bounding box/rectangle) will appear on the video frame to enclose/mark that object
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)), # the class label of the enclosed object appears on the top of the shape
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame) # Output each frame with detection result
    ret, frame = cap.read() # To check if the video has ended. If the video has ended, exit this while loop (stop the model prediction & generate the frame with detection result)

# Must have the section below at the end of the script everytime read/write video using OpenCV
# The section below release the memory it has been allocated for this video
cap.release()
out.release()
cv2.destroyAllWindows()