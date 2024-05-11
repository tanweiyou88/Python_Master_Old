import os

import cv2


img = cv2.imread(os.path.join('D:\Python_Master\Learn_OpenCV_Crop','data','chickens.jpeg')) # read the image file with the specifiled file path
print(img.shape) # to show the size of this image being read, return in (M pixels in height, N pixels in width, P numbers of colour channels)
cv2.imshow('img',img) # to show this image being read.

# crop the image being read by selecting the intervals of the image we want. [Pixel range in height (1 represents the top left corner), Pixel range in width (1 represents the top left corner)]
cropped_img = img[1:148,1:237] # Only crop the top left part
cv2.imshow('cropped_img',cropped_img) # to show the cropped image (top left part).

cropped_center_img = img[117:234,189:379] # Only crop the center part
cv2.imshow('cropped_center_img',cropped_center_img) # to show the cropped image (center part).

cv2.waitKey(0) # wait until a key is pressed before closing the frame