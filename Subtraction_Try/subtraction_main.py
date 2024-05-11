import cv2

video_path = 'D:\Python_Master\Subtraction_Try\WATCH U 1_ch1_20230219100206_20230219100356.mp4'
cap = cv2.VideoCapture(video_path)

subtractor = cv2.createBackgroundSubtractorMOG2(history=120,varThreshold=30)

while True:
    ret, frame = cap.read()

    mask = subtractor.apply(frame)

    cv2.imshow('Frame', frame)
    # cv2.imshow('mask',mask)
    # _, mask = cv2.threshold(mask,120,255,cv2.THRESH_BINARY_INV) # 'img_gray' is the image to be processed using threshold. '120' is the global/absolute threshold that every pixel will follow. For all pixels whose values are below the global threshold, their values will be taken to 0[Black].'255' means for all pixels whose values are above the global threshold, their values will be taken to 255[White].'cv2.THRESH_BINARY_INV' is the threshold method, inverse the black to white & vice versa.
    # Median blur
    # k_size_small = 51 # kernel size. The larger the kernel size, the stronger the blur.
    # img_GaussianBlur_SmallSize = cv2.GaussianBlur(mask,(k_size_small,k_size_small), 3) # Gaussian blur. 'img' is the image we want to blur, (M pixels,N pixels) represents the kernel/mask size
    # img_MedianBlur_SmallSize = cv2.medianBlur(mask,k_size_small) # Median blur. 'img' is the image we want to blur, (M pixels) represents the kernel/mask size
    # cv2.imshow('img_MedianBlur_SmallSize',img_MedianBlur_SmallSize) # show the blurred image using Median Blur.

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image from BGR into Gray colour space
    # cv2.imshow('img_gray',img_gray) # show the image converted into grayscale/gray colour space (pixel value from 0[Black] to 255[white])

    # ret, thresh_inv = cv2.threshold(img_gray,120,255,cv2.THRESH_BINARY_INV) # 'img_gray' is the image to be processed using threshold. '120' is the global/absolute threshold that every pixel will follow. For all pixels whose values are below the global threshold, their values will be taken to 0[Black].'255' means for all pixels whose values are above the global threshold, their values will be taken to 255[White].'cv2.THRESH_BINARY_INV' is the threshold method, inverse the black to white & vice versa.
    # cv2.imshow('thresh_inv',thresh_inv) # 'thresh_inv' is the image processed from 'img_gray' using threshold + binary inverse function

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # contours is a Python list of all the contours(1 countour/border for 1 isolated object/connected boundary) in the image. hierachy stores the coordinates of boundary points for each individual contour


    for cnt in contours: # using a for loop
        # print(cv2.contourArea(cnt)) # show the area of each countour listed in 'contours' above
        ## Build object detector using (1) contour OR (2) bounding box
        if cv2.contourArea(cnt) > 500: # Perform the following tasks if the contour area is larger than 60 (To remove the noise whose contour area is smaller than 60)
            # (1) using contour
            cv2.drawContours(mask, cnt, -1, (0,0,255), 3) # Draw each listed contour on the image called 'img'. (B value,G value,R value) refers to the colour of the contour/border drawn. '3' refers to the thickness of the contour/border drawn.
            # (2) using bounding box
            x1,y1,w,h = cv2.boundingRect(cnt) # Get the bounding box around each listed contour, 1 iteration for 1 listed contour.
            cv2.rectangle(mask, (x1,y1),(x1+w,y1+h), (255,0,0), 3) # Draw the bounding box (rectangle) on the image called 'img' using the (x1,y1) as top left point coordinates of the bounding box and (w,h) as width and height of the bounding box.(B value,G value,R value) refers to the colour of the bounding box drawn. '3' refers to the thickness of the bounding box drawn.

            

    cv2.imshow('mask',mask) # show the original image where contours/bounding boxes are drawn on it.

    # Break the loop if 'q' is pressed
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()