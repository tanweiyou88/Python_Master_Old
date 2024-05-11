
import os

import cv2

## read video
# Concatenate Base directory + folder + file name = image_path='D:\Python_Master\Learn_OpenCV_InputOutputVideo\data\Despicable_Me_2.mp4'
video_path = os.path.join('D:\Python_Master\Learn_OpenCV_InputOutputVideo','data','Despicable_Me_2.mp4')
video = cv2.VideoCapture(video_path) # Read the video file


## visualize video

ret = True
while ret:
    # video.read() return 2 parameters, ret & frame. 
    # Frame is the actual frame of that video at every moment.
    # ret = 1(True) when a frame is read from the video (the video playback not yet ended). ret = 0 when no frame is read from the video (the video playback has reached the end)
    ret, frame = video.read()

    if ret: # when a frame is read from the video (the video playback not yet ended)
        cv2.imshow('frame',frame) # Open a window called 'frame' to show a frame as a picture at the moment the frame is read by video.read(). No audio will be played by the window.
        # cv2.waitKey(40) # Show the single frame for 40 millisecond (25 frames per second)
        cv2.waitKey(20) # Show the single frame for 20 millisecond (50 frames per second, faster than 25 frames per second, so the playback will be smoother)

# Must have the section below at the end of the script everytime read video using OpenCV
# The section below release the memory it has been allocated for this video
video.release()
cv2.destroyAllWindows()


