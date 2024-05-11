"""
Project: Real-time Object Tracking with YOLOv9 and DeepSORT algorithm
Aim: Serves as the complete pipeline to detect and track objects on a video when the video is playing (real-time).
It is optional to save the video with detection information annotated. It is optional to visualize the moving path of all detections on each frame.
Youtube tutorial repository: https://github.com/MuhammadMoinFaisal/YOLOv9-DeepSORT-Object-Tracking. Visit here to get codes for different application requirements (EG: use webcam, choose to draw moving path, choose to detect a specific object class, choose to perform detection without tracking,choose to save the video with detection information annotated...)
YOLOv9 Official repository: https://github.com/WongKinYiu/yolov9
YOLOv9 Pretrained Models Download Official Repository: https://github.com/WongKinYiu/yolov9/releases/tag/v0.1

Complete step of this project:
1) Update the machine learning model (object detector) you want to use in this project 
2) Update the location and name of the video where detections and trackings will be performed on it (if applicable). The video should be located at the same directory of the main file of the project.
3) At the terminal, use cd to switch to the directory where the main file of this project is located (EG: "detect_dual_tracking.py" is the main file for the project performing detection and tracking)
4) At the terminal, implement the codes for different application requirements (EG: use webcam, choose to draw moving path, choose to detect a specific object class, choose to perform detection without tracking,...)
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import math
import torch
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

def initialize_deepsort():
    # Create the Deep SORT configuration object and load settings from the YAML file
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    # Initialize the DeepSort tracker
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        # min_confidence  parameter sets the minimum tracking confidence required for an object detection to be considered in the tracking process
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        #nms_max_overlap specifies the maximum allowed overlap between bounding boxes during non-maximum suppression (NMS)
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        #max_iou_distance parameter defines the maximum intersection-over-union (IoU) distance between object detections
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        # Max_age: If an object's tracking ID is lost (i.e., the object is no longer detected), this parameter determines how many frames the tracker should wait before assigning a new id
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        #nn_budget: It sets the budget for the nearest-neighbor search.
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True
        )

    return deepsort

deepsort = initialize_deepsort()
data_deque = {}
def classNames():
    cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    return cocoClassNames
className = classNames()

def colorLabels(classid): # define the colour of bouding box for different types of objects detected
    if classid == 0: #person
        color = (85, 45, 255)
    elif classid == 2: #car
        color = (222, 82, 175)
    elif classid == 3: #Motorbike
        color = (0, 204, 255)
    elif classid == 5: #Bus
        color = (0,149,255)
    else:
        color = (200, 100,0)
    return tuple(color)

def draw_boxes(frame, bbox_xyxy, draw_trails, identities=None, categories=None, offset=(0,0)):
    height, width, _ = frame.shape # get the height and width of the current frame
    for key in list(data_deque): # track the moving path of the detection
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox_xyxy): # look through each bounding box coordinates
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0] # add offset value to the bounding box coordinate
        y1 += offset[0]
        x2 += offset[0]
        y2 += offset[0]
        center = int((x1+x2)/2), int((y1+y2)/2) # Find the center point of the bounding box
        cat = int(categories[i]) if categories is not None else 0 # get the class ID for each detection (detected object)
        color = colorLabels(cat) # get the colour for the class ID of the detection (EG: if a people is detected, its bounding box will be red)
        #color = [255,0,0]#compute_color_labels(cat)
        id = int(identities[i]) if identities is not  None else 0 # the unique ID of the detection
        # create new buffer for new object
        if id not in data_deque:
          data_deque[id] = deque(maxlen= 64)
        data_deque[id].appendleft(center)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # the characteristics/appearance of the bounding box
        name = className[cat] # the class name according to its class ID
        label = str(id) + ":" + name # display its unique ID & class name above its bounding box
        text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0] # the characteristics/appearance of the text for the unique ID & class name above its bounding box
        c2 = x1 + text_size[0], y1 - text_size[1] - 3 # define the bottom right corner coordinates of the rectangle above the bounding box
        cv2.rectangle(frame, (x1, y1), c2, color, -1) # create the rectangle above the bounding box 
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA) # insert the unique ID & class name of the detection in the rectangle above its bounding box 
        cv2.circle(frame,center, 2, (0,255,0), cv2.FILLED) # create a circle at the center of each detection
        if draw_trails: # if the user wants to draw a tail to visualize the moving path of the detection
              # draw trail
              for i in range(1, len(data_deque[id])):
                  # check if on buffer value is none
                  if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                      continue
                  # generate dynamic thickness of trails
                  thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                  # draw trails
                  cv2.line(frame, data_deque[id][i - 1], data_deque[id][i], color, thickness)    
    return frame

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        draw_trails = False,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            pred = pred[0][1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            ims = im0.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                xywh_bboxs = []
                confs = []
                oids = []
                outputs = []
                # Write results
                for *xyxy, conf, cls in reversed(det): # Outputs provided by YOLOv9: xyxy is the coordinates of the top left corner & bottom right corner of a bounding box in the form of tensor, conf is the confidence score of the detected object, cls is the class ID of the object
                    x1, y1, x2, y2 = xyxy # stores the coordinates of the top left corner & bottom right corner of a bounding box in the form of tensor in 4 respective variables
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert the coordinates of the top left corner & bottom right corner of a bounding box from tensor into interger form.
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2) # Find the Center Coordinates of the bounding box for each detected object
                    bbox_width = abs(x1-x2) # Find the width of the boundng box
                    bbox_height = abs(y1-y2) # Find the height of the boundng box
                    xcycwh = [cx, cy, bbox_width, bbox_height] # store the information according to the input format of DeepSORT algorithm
                    xywh_bboxs.append(xcycwh) # append the information (center coordinates, width, and height) of this bounding box
                    conf = math.ceil(conf*100)/100
                    confs.append(conf) # append the information (confidence score) of this bounding box
                    classNameInt = int(cls) # convert the class ID from tensor into integer format
                    oids.append(classNameInt) # append the information (class ID) of this bounding box
                xywhs = torch.tensor(xywh_bboxs) # convert the information (center coordinates, width, and height) of this bounding box into tensor format
                confss = torch.tensor(confs) # convert the information (confidence score) of this bounding box into tensor format
                outputs = deepsort.update(xywhs, confss, oids, ims) # apply DeepSORT to track (assign an unique ID to) every detection on the frame. imd represents the current video frame. DeepSORT receive 4 inputs (the width, height, and center coordinates of a bounding box).
                if len(outputs) > 0: # if there is a detection on the frame, extract the DeepSORT outputs for that detection
                    bbox_xyxy = outputs[:, :4] # top left corner and bottom right corner coordinates of the bounding box (identified by the object detector at earlier stage)
                    identities = outputs[:, -2] # the unique ID assigned by DeepSORT to the detection (this parameter is the purpose we apply DeepSORT, which is the important information to realize object tracking)
                    object_id = outputs[:, -1] # the class ID of the bounding box (identified by the object detector at earlier stage)
                    draw_boxes(ims, bbox_xyxy, draw_trails, identities, object_id) # draw the bounding box with unique and class IDs annotated around the detection on the current frame. Also draw the tail of that detection to see its moving path

            # Stream results
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), ims.shape[1], ims.shape[0])
                cv2.imshow(str(p), ims)
                cv2.waitKey(1)  # 1 millisecond
            # Save results (image with detections)
            if save_img:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, ims.shape[1], ims.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(ims)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--draw-trails', action='store_true', help='do not drawtrails')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))



if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


'''
Example of codes can be executed at the Terminal to implement this project:
Compulsory: Use cd to switch to the directory where the main file of this project is located (EG: "detect_dual_tracking.py" is the main file for the project performing detection and tracking)
Option 1) 
Implement: python detect_dual_tracking.py --weight 'yolov9-c.pt' --source 'testwalk.mp4' --device 'cpu' --view-img
#  this codes means run "detect_dual_tracking.py", with 'yolov9-c.pt' as the object detector, to detect objects (objects of all classes that 
the object detector is trained to detect) on the video called 'testwalk.mp4' located on the same directory, using CPU only (without GPU). Then, show each video frame as an image to visualize the information of all detections on that frame.
The video with detection information annotated will not be saved.

Option 2) 
Implement: python detect_dual_tracking.py --weight 'yolov9-c.pt' --source 'testwalk.mp4' --device 'cpu' --view-img --draw-trails --class 0 
#  this codes means run "detect_dual_tracking.py", with 'yolov9-c.pt' as the object detector, to detect specific objects only (--class 0 is defined as people in this project, so detect people only. if --class is not mentioned, then objects of all classes that 
the object detector is trained to detect will be detected) on the video called 'testwalk.mp4' located on the same directory, using CPU only (without GPU). Then, show each video frame as an image to visualize the information of all detections on that frame.
Also draw a tail for each detection to visualize its moving path (because --draw-trails is mentioned here). The video with detection information annotated will not be saved.

Option 3)
Implement: python detect_dual_tracking.py --weight 'yolov9-c.pt' --source 'testwalk.mp4' --device 'cpu' --view-img --draw-trails --class 0 --project 'Detection_and_tracking_results' --name 'testwalk_output'
#  this codes means run "detect_dual_tracking.py", with 'yolov9-c.pt' as the object detector, to detect specific objects only (--class 0 is defined as people in this project, so detect people only. if --class is not mentioned, then objects of all classes that 
the object detector is trained to detect will be detected) on the video called 'testwalk.mp4' located on the same directory, using CPU only (without GPU). Then, show each video frame as an image to visualize the information of all detections on that frame.
Also draw a tail for each detection to visualize its moving path (because --draw-trails is mentioned here). Since [--project 'Detection_and_tracking_results' --name 'Detection_and_tracking_results'] is mentioned, the video with detection information annotated 
will be saved in the same directory of the main file, in the subfolder named 'Detection_and_tracking_results, in the subsubfolder named 'testwalk_output'. 
'''

