
import os

import cv2

## read image
# Concatenate Base directory + folder + file name = image_path='D:\Python_Master\Learn_OpenCV_InputOutputImage\data\cat.jpeg'
image_path = os.path.join('D:\Python_Master\Learn_OpenCV_InputOutputImage','data','cat.jpeg')
img = cv2.imread(image_path)

## write image
# Write/paste the file called "img" to the path/directory stated by os.path.join with filename of 'cat_Duplicated.jpeg'
cv2.imwrite(os.path.join('D:\Python_Master\Learn_OpenCV_InputOutputImage','data','cat_Duplicated.jpeg'),img)

## visualize image
cv2.imshow('Image_cat',img) # Open a frame with name of 'Image_cat' that show the content of 'img'
cv2.waitKey(0) # wait until a key is pressed before closing the frame
# cv2.waitKey(2000) # wait for 2 seconds before closing the frame



