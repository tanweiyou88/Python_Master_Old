import numpy as np
import cv2

# Concept: use a range of Hue(H) channel/component of HSV colour space to specify the colour we want. Then we want all the pixels (S & V components) that considered within the H channel range.
def get_limits(color): # Define a function called 'get_limits' that accept a parameter called 'color'. The 'color' stores the BGR values which you want to convert to HSV

    c = np.uint8([[color]]) # here insert the BGR values which you want to convert to HSV
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit