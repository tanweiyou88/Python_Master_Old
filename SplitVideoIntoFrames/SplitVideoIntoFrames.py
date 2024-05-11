import os
import cv2
 


capture = cv2.VideoCapture(os.path.join('D:\Python_Master','SplitVideoIntoFrames','2e865e3eda749c13bce63beeacf86c66.mp4')) # Absolute path to the video
 
frameNr = 0
 
while (True):
 
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(f'D:/Python_Master/SplitVideoIntoFrames/Splitted_frames/frame_{frameNr}.jpg', frame)
 
    else:
        print('Error')
        break
 
    frameNr = frameNr+1
 
capture.release()