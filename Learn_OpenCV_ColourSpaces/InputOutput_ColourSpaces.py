import os

import cv2

img = cv2.imread(os.path.join('D:\Python_Master\Learn_OpenCV_ColourSpaces','data','Red_Bird.jpg')) # 'img' is the image being read from the specified file path.
cv2.imshow('img',img) # show the image being read

## Convert from BGR to RGB colour space
# when read an image using OpenCV, the image will be in the Green,Red,Blue (GRB) colour space 
img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # to convert an image which is in a given colored space to another color space. 'img' is the image to be converted, 'cv2.COLOR_BGR2RGB' convert the image from BGR (3 channels) to RGB (3 channels) colour space. Red switched to blue colour, and vice versa.
cv2.imshow('img_RGB',img_RGB)

## Convert from BGR to Gray colour space
img_Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # to convert an image which is in a given colored space to another color space. 'img' is the image to be converted, 'cv2.COLOR_BGR2GRAY' convert the image from BGR (3 channels) to Gray (1 channel) colour space.
cv2.imshow('img_Gray',img_Gray)

## Convert from BGR to HSV colour space
img_HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # to convert an image which is in a given colored space to another color space. 'img' is the image to be converted, 'cv2.COLOR_BGR2HSV' convert the image from BGR (3 channels) to HSV (3 channels) colour space.
cv2.imshow('img_HSV',img_HSV)


cv2.waitKey(0) # wait until a key is pressed before closing the frame