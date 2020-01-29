import cv2 as cv
import MyImageUtils
import ObjectPredictation as op
import numpy as np

# READ IMAGE
image = cv.imread('../images/ContainerLeftSideView.jpg')
imCopy = image.copy()
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# IMAGE THRESHOLD
threshImg = MyImageUtils.getThresholdedImage(gray)
edges_high_thresh = cv.Canny(gray, 80, 140)
# CONTOUR IMAGE
contours, hierarchy = cv.findContours(edges_high_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# Iterate through each contour
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    # Define View Angle TODO
    if True:
        # Define Have Container By View Type TODO
        if op.isHaveContainer('Left'):
            # Define Object Type Before Draw Rectangle TODO
            if w > 200 and h > 200:
                cv.rectangle(imCopy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv.putText(imCopy, 'Test', (x, y + 5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        else:
            cv.putText(imCopy, "don't have container", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5, cv.LINE_AA)
# Show Window
images = np.hstack((image, edges_high_thresh, imCopy))
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', images)
cv.waitKey(0)
cv.destroyAllWindows()
