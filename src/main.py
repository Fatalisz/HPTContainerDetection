import cv2 as cv
import myImageUtils
import numpy as np

# READ IMAGE
image = cv.imread('../images/gen.jpg')
imCopy = image.copy()
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# IMAGE THRESHOLD
threshImg = myImageUtils.getThresholdedImage(gray)
# CONTOUR IMAGE
contours, hierarchy = cv.findContours(threshImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# Iterate through each contour
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(imCopy, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv.imshow('image with contour', imCopy)
cv.imshow('image origin', image)
cv.imshow('image threshold', threshImg)
cv.waitKey(0)
cv.destroyAllWindows()
