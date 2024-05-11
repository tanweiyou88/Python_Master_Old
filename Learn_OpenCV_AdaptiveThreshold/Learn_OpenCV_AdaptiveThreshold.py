import os

import cv2

img = cv2.imread(os.path.join('D:\Python_Master\Learn_OpenCV_AdaptiveThreshold','data','Writing.jpeg'))  # read the image file with the specifiled file path
cv2.imshow('img',img) # show the original image

## Threshold using single/global/absolute threshold
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image from BGR into Gray colour space
ret, thresh_global = cv2.threshold(img_gray,110,255,cv2.THRESH_BINARY) # 'img_gray' is the image to be processed using threshold, it must in grayscale. '110' is the global/absolute threshold that every pixel will follow. For all pixels whose values are below the global threshold, their values will be taken to 0[Black].'255' means for all pixels whose values are above the global threshold, their values will be taken to 255[White].'cv2.THRESH_BINARY' is the threshold method.
cv2.imshow('thresh_global',thresh_global) # 'thresh' is the image processed from 'img_gray' using global threshold 

## Threshold using adaptive/multiple threshold
thresh_adaptive = cv2.adaptiveThreshold(img_gray,25,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,10) # through adaptive threshold function, OpenCV will figure the threshold out by itself.'img_gray' is the image to be processed using threshold, it must in grayscale.'255' means for all the pixels whose values are higher than the given threshold computed by OpenCV, their values will be taken to 255[White].The remaining parameters also affect the way of applying adaptive threshold.
cv2.imshow('thresh_adaptive',thresh_adaptive) # 'thresh_adaptive' is the image processed from 'img_gray' using adaptive threshold 

cv2.waitKey(0) # wait until a key is pressed before closing the frame/window