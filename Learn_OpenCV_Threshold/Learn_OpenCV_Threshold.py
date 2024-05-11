import os

import cv2

img = cv2.imread(os.path.join('D:\Python_Master\Learn_OpenCV_Threshold','data','Bear.jpg')) # read the image file with the specifiled file path
cv2.imshow('img',img) # show the original image

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert the image from BGR into Gray colour space
cv2.imshow('img_gray',img_gray) # show the image converted into grayscale/gray colour space (pixel value from 0[Black] to 255[white])

## Threshold
ret, thresh = cv2.threshold(img_gray,100,255,cv2.THRESH_BINARY) # 'img_gray' is the image to be processed using threshold. '100' is the global/absolute threshold that every pixel will follow. For all pixels whose values are below the global threshold, their values will be taken to 0[Black].'255' means for all pixels whose values are above the global threshold, their values will be taken to 255[White].'cv2.THRESH_BINARY' is the threshold method.
cv2.imshow('thresh',thresh) # 'thresh' is the image processed from 'img_gray' using threshold 

## Improved the image processed with threshold using blurring
thresh_blurred = cv2.blur(thresh, (10,10)) # Perform classical blur on 'thresh' image file with kernel size of (10,10)
ret_improved, thresh_improved = cv2.threshold(thresh_blurred,100,255,cv2.THRESH_BINARY)
cv2.imshow('thresh_improved',thresh_improved) # 'thresh_improved' is the image processed from 'thresh_blurred' using threshold 

cv2.waitKey(0) # wait until a key is pressed before closing the frame/window