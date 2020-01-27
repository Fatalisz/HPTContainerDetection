import cv2 as cv
import myImageUtils
import ObjectPredictation as op
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
    # Define Object Type Before Draw Rectangle
    if op.isHaveContainer():
        cv.rectangle(imCopy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(imCopy, 'Test', (x, y + 5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
    else:
        cv.putText(imCopy, "don't have container", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5, cv.LINE_AA)
# Show Window
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', imCopy)
cv.waitKey(0)
cv.destroyAllWindows()
