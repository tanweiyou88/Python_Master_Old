import os

import cv2


img = cv2.imread(os.path.join('D:\Python_Master\Learn_OpenCV_Drawing','data','white board.jpeg')) # read the image file with the specifiled file path
print(img.shape) # show the size of the image called 'img'

## Line
cv2.line(img,(100,150),(200,300),(255,0,0),3) # 'img' is the image which we will draw on top of it. (x-coordinate of starting point, y-coordinate of starting point) is the starting point coordinates, (x-coordinate of ending point, y-coordinate of ending point) is the ending point coordinates.(B value,G value,R value) refers to the colour of the line drawn. '3' refers to the thickness of the line drawn.

## Rectangle
cv2.rectangle(img,(200,150),(250,200),(0,0,255),15) # 'img' is the image which we will draw on top of it. (x-coordinate of top left point, y-coordinate of top left point) is the top left point, (x-coordinate of bottom right point, y-coordinate of bottom right point) is the bottom right point coordinates.(B value,G value,R value) refers to the colour of the rectangle drawn. '5' refers to the thickness of the rectangle drawn.
cv2.rectangle(img,(300,150),(350,200),(0,0,255),-1) # 'img' is the image which we will draw on top of it. (x-coordinate of top left point, y-coordinate of top left point) is the top left point, (x-coordinate of bottom right point, y-coordinate of bottom right point) is the bottom right point coordinates.(B value,G value,R value) refers to the colour of the rectangle drawn. '-1' refers the rectangle drawn is filled.

## Circle
cv2.circle(img, (400,100), 40, (0,255,255), 10) # 'img' is the image which we will draw on top of it. (x-coordinate of circle center, y-coordinate of circle center) is the coordinates of the circle center. '10' refers to the radius of the circle. (B value,G value,R value) refers to the colour of the circle drawn. '10' refers to the thickness of the circle drawn.

## Text
cv2.putText(img, 'Hey you!', (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 4) # 'img' is the image which we will write on top of it. 'Hey you!' is the text we want to write. (x-coordinate, y-coordinate) is the coordinate we write. 'cv2.FONT_HERSHEY_SIMPLEX' is the font type. '1' refers to the size of the text. (B value,G value,R value) refers to the colour of the text. '4' refers to the thickness of the text.

cv2.imshow('img',img) # show the image of white board. This sentence must be after the sentences that draw drawings so that the window will show drawings on the white board image.
cv2.waitKey(0) # wait until a key is pressed before closing the frame/window
