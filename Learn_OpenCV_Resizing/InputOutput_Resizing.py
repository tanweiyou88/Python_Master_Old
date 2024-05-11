import os

import cv2

img = cv2.imread(os.path.join('D:\Python_Master\Learn_OpenCV_Resizing','data','chickens.jpeg')) # read the image file with the specifiled file path
print(img.shape) # to show the size of this image being read, return in (M pixels in height, N pixels in width, P numbers of colour channels)
cv2.imshow('img',img) # to show this image being read.

# to resize the image being read. 'img' represents the image being read, (A pixels in width, B pixels in height) represents the new size for this image
# the parameter (237,148) resize the image into 0.5 of its original size.
resized_img = cv2.resize(img,(237,148)) 
print(resized_img.shape) # to show the size of the resized image, return in (M pixels in height, N pixels in width, P numbers of colour channels)
cv2.imshow('resized_img',resized_img) # to show the resized image.


cv2.waitKey(0) # wait until a key is pressed before closing the frame/window