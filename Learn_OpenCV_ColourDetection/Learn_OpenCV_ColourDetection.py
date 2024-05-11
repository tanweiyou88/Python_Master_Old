import cv2
from PIL import Image
from util import get_limits

yellow = [0,255,255] # Define yellow in BGR colour space

## read webcam
webcam = cv2.VideoCapture(0) # The 0 represents the number/ID of the webcam I want to access from my computer. If the computer only has single webcam, then its number will be 0.

## visualize webcam

while True: # because when reading webcam, we always have new frames to read. So using while True to continue the loop.
    ret, frame = webcam.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert the frame, that is read from the webcam at the moment, from BGR to HSV colour space.
    
    lowerLimit, upperLimit = get_limits(color=yellow) # Call the 'get_limits' function and set 'yellow' as its parameter, 'color'. This function will return the lower and upper limits of H component in HSV colour space that are considered as yellow colour.

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit) # to get a mask from all the pixels that belong to the colour we want to detect. Each mask stores the location of all the pixels containing the infomration(colour) we want & the pixels' value are set to 255[White] respectively. The input of this function is an image we want to process with HSV as colour space, followed by the lower and upper limits of H component for the colour we want to detect.
    mask_ = Image.fromarray(mask) # convert an image/frame from numpy array format (which is OpenCV representation for images) into pillow
    bbox = mask_.getbbox() # get the bounding box of the mask
    print(bbox) # print the bounding box. It shows bbox=NONE when no mask is obtained (colour we want is not detected). Else, 4 values [x-coordinate of top left point, y-coordinate of top left point, x-coordinate of bottom right point, y-coordinate of bottom right point] will be returned.
    if bbox is not None:
        x1, y1, x2, y2 = bbox # get locations [x-coordinate of top left point, y-coordinate of top left point, x-coordinate of bottom right point, y-coordinate of bottom right point] of the bounding box
        frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),5) # Draw the bounding box on the 'frame' and stores the drawn 'frame' to 'frame' again. (x-coordinate of top left point, y-coordinate of top left point) is the top left point, (x-coordinate of bottom right point, y-coordinate of bottom right point) is the bottom right point coordinates.(B value,G value,R value) refers to the colour of the rectangle drawn. '5' refers to the thickness of the rectangle drawn.


    # cv2.imshow('frame',mask) # Open a window called 'frame' to show the mask obtained from the image called 'hsvImage'.
    cv2.imshow('frame',frame) # Open a window called 'frame' to show the frame where bounding box is drawn on it, when the colour we want is detected.
    if (cv2.waitKey(40) & 0xFF) == ord('q') : # it waits for 40 milliseconds for key press (counted from a frame is shown by the window) and checks whether the pressed key is q. Reference: https://stackoverflow.com/questions/67356538/can-anyone-tell-me-why-the-code-below-used-waitkey20-and-what-does-this-0xff#:~:text=The%200xFF%20does%20not%20belong%20to%20the%20%3D%3D,be%20%28cv2.waitKey%20%2820%29%20%26%200xFF%29%20%3D%3D%20ord%20%28%27q%27%29.
        break # means we are going to wait 40 milliseconds after the window showed a frame and once the user presses the small letter 'Q' on keyboard, we are going to exit this while True loop


webcam.release()
cv2.destroyAllWindows()
