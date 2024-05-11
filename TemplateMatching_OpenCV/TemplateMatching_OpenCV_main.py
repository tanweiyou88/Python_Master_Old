#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

#https://youtu.be/P5FTEryiTl4

"""
OBJECT DETECTION WITH TEMPLATES

Need a source image and a template image.
The template image T is slided over the source image (as in 2D convolution), 
and the program tries to find matches using statistics.
Several comparison methods are implemented in OpenCV.
It returns a grayscale image, where each pixel denotes how much does the 
neighbourhood of that pixel match with template.

Once you got the result, you can use cv2.minMaxLoc() function 
to find where is the maximum/minimum value. Take it as the top-left corner of the 
rectangle and take (w,h) as width and height of the rectangle. 
That rectangle can be drawn on the region of matched template.
"""
### Template matching, single object in an image.
#Multiple methods to see which one works best. 

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img_rgb = cv2.imread('D:\Python_Master\TemplateMatching_OpenCV\ezgif-frame-177.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('D:\Python_Master\TemplateMatching_OpenCV\RawPic1_TopLeft_Template.png', 0)
# h, w = template.shape[::] 

# cv2.imshow("Source", img_rgb)
# cv2.imshow("Template", template)

# #methods available: ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
# #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
# # For TM_SQDIFF, Good match yields minimum value; bad match yields large values
# # For all others it is exactly opposite, max value = good fit.
# plt.imshow(res, cmap='gray')

# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv2.rectangle(img_gray, top_left, bottom_right, 255, 2)  #White rectangle with thickness 2. 

# cv2.imshow("Matched image", img_gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

       
### Template matching - multiple objects

#For multiple occurances, cv2.minMaxLoc() won’t give all the locations
#So we need to set a threshold
    



# import os
# import cv2

# img = cv2.imread('D:\Python_Master\TemplateMatching_OpenCV\RawPic1-frame-001.jpg') # read the image file with the specifiled file path
# cv2.imshow('img',img) # show the original image

# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert the image from BGR into Gray colour space
# cv2.imshow('img_gray',img_gray) # show the image converted into grayscale/gray colour space (pixel value from 0[Black] to 255[white])

# ## Threshold
# ret, thresh = cv2.threshold(img_gray,70,255,cv2.THRESH_BINARY) # 'img_gray' is the image to be processed using threshold. '100' is the global/absolute threshold that every pixel will follow. For all pixels whose values are below the global threshold, their values will be taken to 0[Black].'255' means for all pixels whose values are above the global threshold, their values will be taken to 255[White].'cv2.THRESH_BINARY' is the threshold method.
# cv2.imshow('thresh',thresh) # 'thresh' is the image processed from 'img_gray' using threshold 

# ## Improved the image processed with threshold using blurring
# thresh_blurred = cv2.blur(thresh, (10,10)) # Perform classical blur on 'thresh' image file with kernel size of (10,10)
# # ret_improved, thresh_improved = cv2.threshold(thresh_blurred,100,255,cv2.THRESH_BINARY)
# # cv2.imshow('thresh_improved',thresh_improved) # 'thresh_improved' is the image processed from 'thresh_blurred' using threshold 

# cv2.waitKey(0) # wait until a key is pressed before closing the frame/window


















import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('D:\Python_Master\TemplateMatching_OpenCV\RawPic1-frame-001.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('D:\Python_Master\TemplateMatching_OpenCV\Template5.png',0)
h, w = template.shape[::]

# cv2.imshow("Source", img_rgb)
cv2.imshow("Template", template)

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
plt.imshow(res, cmap='gray')

threshold = 0.6 #Pick only values above 0.8. For TM_CCOEFF_NORMED, larger values = good fit.

loc = np.where( res >= threshold)  
#Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

#Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
#then the second item and then third, etc. 

for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
    #Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)  #Red rectangles with thickness 2. 

#cv2.imwrite('images/template_matched.jpg', img_rgb)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()




# import cv2
 
# # Read the original image
# img = cv2.imread('D:\Python_Master\TemplateMatching_OpenCV\RawPic1-frame-001.jpg') 
# # Display original image
# cv2.imshow('Original', img)
 
# # Convert to graycsale
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Blur the image for better edge detection
# img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
# # # Sobel Edge Detection
# # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
# # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
# # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# # # Display Sobel Edge Detection Images
# # cv2.imshow('Sobel X', sobelx)
# # cv2.waitKey(0)
# # cv2.imshow('Sobel Y', sobely)
# # cv2.waitKey(0)
# # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
# # cv2.waitKey(0)
 
# # Canny Edge Detection
# edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200) # Canny Edge Detection
# # Display Canny Edge Detection Image
# cv2.imshow('Canny Edge Detection', edges)

# contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # contours is a Python list of all the contours(1 countour/border for 1 isolated object/connected boundary) in the image. hierachy stores the coordinates of boundary points for each individual contour


# for cnt in contours: # using a for loop
#     print(cv2.contourArea(cnt)) # show the area of each countour listed in 'contours' above
#     ## Build object detector using (1) contour OR (2) bounding box
#     if cv2.contourArea(cnt): # Perform the following tasks if the contour area is larger than 60 (To remove the noise whose contour area is smaller than 60)
#         # (1) using contour
#         cv2.drawContours(img, cnt, -1, (0,0,255), 3) # Draw each listed contour on the image called 'img'. (B value,G value,R value) refers to the colour of the contour/border drawn. '3' refers to the thickness of the contour/border drawn.
#         # (2) using bounding box
#         x1,y1,w,h = cv2.boundingRect(cnt) # Get the bounding box around each listed contour, 1 iteration for 1 listed contour.
#         # cv2.rectangle(img, (x1,y1),(x1+w,y1+h), (255,0,0), 3) # Draw the bounding box (rectangle) on the image called 'img' using the (x1,y1) as top left point coordinates of the bounding box and (w,h) as width and height of the bounding box.(B value,G value,R value) refers to the colour of the bounding box drawn. '3' refers to the thickness of the bounding box drawn.


# cv2.imshow('img',img) # show the original image where contours/bounding boxes are drawn on it.

# cv2.waitKey(0)
 
# cv2.destroyAllWindows()