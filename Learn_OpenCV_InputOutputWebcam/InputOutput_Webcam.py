import cv2

## read webcam
webcam = cv2.VideoCapture(0) # The 0 represents the number/ID of the webcam I want to access from my computer. If the computer only has single webcam, then its number will be 0.

## visualize webcam

while True: # because when reading webcam, we always have new frames to read. So using while True to continue the loop.
    ret, frame = webcam.read()

    cv2.imshow('frame',frame) # Open a window called 'frame' to show a frame that is read from the webcam at the moment
    if (cv2.waitKey(40) & 0xFF) == ord('q') : # it waits for 40 milliseconds for key press (counted from a frame is shown by the window) and checks whether the pressed key is q. Reference: https://stackoverflow.com/questions/67356538/can-anyone-tell-me-why-the-code-below-used-waitkey20-and-what-does-this-0xff#:~:text=The%200xFF%20does%20not%20belong%20to%20the%20%3D%3D,be%20%28cv2.waitKey%20%2820%29%20%26%200xFF%29%20%3D%3D%20ord%20%28%27q%27%29.
        break # means we are going to wait 40 milliseconds after the window showed a frame and once the user presses the small letter 'Q' on keyboard, we are going to exit this while True loop


webcam.release()
cv2.destroyAllWindows()
