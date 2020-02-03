import cv2 as cv
import MyImageUtils
import ObjectPredictation as op
import numpy as np

# READ IMAGE
image = cv.imread('../images/Top/IMG_4208.jpg')
imCopy = image.copy()
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# IMAGE THRESHOLD
#threshImg = MyImageUtils.getThresholdedImage(gray)
#result, threshImg = cv.threshold(gray, 0, 120, cv.THRESH_OTSU|cv.THRESH_BINARY_INV)
result, threshImg = cv.threshold(gray, 0, 120, cv.THRESH_BINARY)
# EDGE DETECTION
#edges_high_thresh = cv.Canny(gray, 100, 200)
#DILATION
rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 3))
dilation = cv.dilate(threshImg, rect_kernel, iterations = 1)
cv.imshow('dilation', dilation)
# CONTOUR IMAGE
contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# Iterate through each contour
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    # Define View Angle TODO
    if True:
        # Define Have Container By View Type TODO
        if op.isHaveContainer('Left'):
            # Define Object Type Before Draw Rectangle TODO
            #if w > 200 and h > 200:
            cv.rectangle(imCopy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(imCopy, 'Test', (x, y + 5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        else:
            cv.putText(imCopy, "don't have container", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5, cv.LINE_AA)
# Show Window
images = np.hstack((gray, dilation))
images2 = np.hstack((image, imCopy))
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', images)
cv.namedWindow('image2', cv.WINDOW_NORMAL)
cv.imshow('image2', images2)
cv.waitKey(0)
cv.destroyAllWindows()
