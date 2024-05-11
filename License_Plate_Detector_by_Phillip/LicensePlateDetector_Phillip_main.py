"""
Project: License Plate Detector, based on YOLOv8
Aim: To train a machine learning model based on YOLOv8 to detect license plates
# YOLOv8 official repository: https://github.com/ultralytics/ultralytics/blob/main/README.md
# License plate detector Youtube tutorial repository: https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide/blob/master/local_env/predict_video.py
# Dataset repository: https://github.com/jonG312/YOLOv8-Vehicle-Plate-Recognition/blob/main/README.md, folder name on local environment "Car_and_License_Plate_from_jonG312_GitHub"
# One of the websites to download public dataset: https://storage.googleapis.com/openimages/web/index.html
# One of the websites to annotate all the images collected for training and testing: https://www.cvat.ai/ 
"""

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch. YOLOv8 Nano model is selected. 
#model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="D:\Python_Master\License_Plate_Detector_by_Phillip\config.yaml", epochs=400)  # train the model. The higher the number of epoch, the deeper the training, the better the machine learning model can predict, the longer the training time is required.
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format


