import os

import cv2


img = cv2.imread(os.path.join('D:\Python_Master\Learn_OpenCV_EdgeDetection','data','spinwheel.jpg')) # read the image file with the specifiled file path
cv2.imshow('img',img) # show the original image

## Edge detection
img_edge = cv2.Canny(img, 100, 200) # Edge detector, using Canny edge detector. 'img' is the image whose edge will be detected, 'img_edge' is the edge image. The 2 numerical parameters affects the edge detection performance.
cv2.imshow('img_edge',img_edge) # show the edge image

## Edge dilation
import numpy as np

thinner_size = 3
thicker_size = 5
img_edge_dilate_thicker = cv2.dilate(img_edge, np.ones((thicker_size,thicker_size),dtype=np.int8)) # Perform edge dilation on the edge image called 'img_edge'. 'thicker_size'in (thicker_size,thicker_size) represents the thickness of the edge to dilate to
cv2.imshow('img_edge_dilate_thicker',img_edge_dilate_thicker) # show the edge image whose edge is dilated thicker
img_edge_dilate_thinner = cv2.dilate(img_edge, np.ones((thinner_size,thinner_size),dtype=np.int8)) # Perform edge dilation on the edge image called 'img_edge'.
cv2.imshow('img_edge_dilate_thinner',img_edge_dilate_thinner) # show the edge image whose edge is dilated thinner

## Edge errode = opposite of edge dilation
img_edge_errode_thicker = cv2.erode(img_edge_dilate_thicker, np.ones((thicker_size,thicker_size),dtype=np.int8)) # Perform edge errode on the dilated edge image called 'img_edge_dilate_thicker'.
cv2.imshow('img_edge_errode_thicker',img_edge_errode_thicker) # show the edge image whose edge is erroded

cv2.waitKey(0) # wait until a key is pressed before closing the frame/window