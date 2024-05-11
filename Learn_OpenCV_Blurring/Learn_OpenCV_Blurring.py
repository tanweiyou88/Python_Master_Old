import os

import cv2

img = cv2.imread(os.path.join('D:\Python_Master\Learn_OpenCV_Blurring','data','Mona_Lisa.jpeg'))  # read the image file with the specifiled file path
cv2.imshow('img',img) # show the original image

## Classical blur
k_size_small = 7 # kernel size. The larger the kernel size, the stronger the blur.
img_ClassicBlur_SmallSize = cv2.blur(img,(k_size_small,k_size_small)) # classical blur. 'img' is the image we want to blur, (M pixels,N pixels) represents the kernel/mask size
cv2.imshow('img_ClassicBlur_SmallSize',img_ClassicBlur_SmallSize) # show the blurred image using classical blur with small kernel size.

k_size_big = 70 
img_ClassicBlur_BigSize = cv2.blur(img,(k_size_big,k_size_big))  # Blur the image with bigger kernel size using classical blur function.
cv2.imshow('img_ClassicBlur_BigSize',img_ClassicBlur_BigSize) # show the blurred image using classical blur with big kernel size.

## Gaussian blur
img_GaussianBlur_SmallSize = cv2.GaussianBlur(img,(k_size_small,k_size_small), 3) # Gaussian blur. 'img' is the image we want to blur, (M pixels,N pixels) represents the kernel/mask size
cv2.imshow('img_GaussianBlur_SmallSize',img_GaussianBlur_SmallSize) # show the blurred image using Gaussian Blur.

## Median blur
img_MedianBlur_SmallSize = cv2.medianBlur(img,k_size_small) # Median blur. 'img' is the image we want to blur, (M pixels) represents the kernel/mask size
cv2.imshow('img_MedianBlur_SmallSize',img_MedianBlur_SmallSize) # show the blurred image using Median Blur.


cv2.waitKey(0)  # wait until a key is pressed before closing the frame/window
