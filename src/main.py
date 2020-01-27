import cv2 as cv
import myImageUtils
import numpy as np

# READ IMAGE
image = cv.imread('../images/gen.jpg', cv.IMREAD_GRAYSCALE)
# IMAGE THRESHOLD
threshImg = myImageUtils.getThresholdedImage(image)

# CONTOUR IMAGE
# contours, hierarchy = cv.findContours(threshImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# for i in range(len(contours)):
#     cv.drawContours(threshImg, contours, i, (0, 255, 0), 3)

# FIND BLOB WITH SIMPLE BLOB
# detector = cv.SimpleBlobDetector()
# keypoints = detector.detect(threshImg)
# im_with_keypoints = cv.drawKeypoints(threshImg, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', threshImg)
cv.waitKey(0)
cv.destroyAllWindows()
