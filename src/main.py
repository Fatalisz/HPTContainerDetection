import cv2 as cv
import myImageUtils

# READ IMAGE
image = cv.imread('../images/gen.jpg', cv.IMREAD_GRAYSCALE)
# IMAGE THRESHOLD
threshImg = myImageUtils.getThresholdedImage(image)
# CONTOUR IMAGE
contours, hierarchy = cv.findContours(threshImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cv.drawContours(threshImg, contours, i, (0, 255, 0), 3)
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', threshImg)
cv.waitKey(0)
cv.destroyAllWindows()
