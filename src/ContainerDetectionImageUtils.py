import os
from datetime import datetime

import ContainerDetectionConstant as const
import matplotlib.patches as mpatches
from CustomImageClass import CustomImageClass
from scipy import ndimage as ndi
from skimage import filters, io, img_as_ubyte
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.measure import label, regionprops
from skimage.morphology import closing, rectangle
from skimage.restoration import denoise_tv_chambolle


# PREPARE IMAGE FOR SEGMENTATION
def preProcessImage(image):
    grayImage = rgb2gray(image)
    # DENOISE IMAGE
    denoiseImage = denoise_tv_chambolle(grayImage)
    # CROP INTEREST PART FOR THRESHOLD
    minInterestHeight, maxInterestHeight, minInterestWidth, maxInterestWidth = getInterestCroppedArea(denoiseImage)
    croppedInputImage = denoiseImage[minInterestHeight:maxInterestHeight, minInterestWidth:maxInterestWidth]
    # fig, ax = plt.subplots(figsize=(4, 3))
    # ax.imshow(croppedInputImage, cmap=plt.cm.gray)
    # ax.set_title('input')
    # ax.axis('off')
    # TEST THRESHOLD
    # fig, ax = try_all_threshold(croppedInputImage, figsize=(10, 8), verbose=False)
    # plt.show()
    # GET THRESHOLD IMAGE
    thresh = filters.threshold_yen(croppedInputImage)
    binaryImage = croppedInputImage > thresh
    # fig, ax = plt.subplots(figsize=(4, 3))
    # ax.imshow(binaryImage, cmap=plt.cm.gray)
    # ax.set_title('binary')
    # ax.axis('off')
    # EDGE DETECTION
    can = canny(binaryImage)
    # fig, ax = plt.subplots(figsize=(4, 3))
    # ax.imshow(can, cmap=plt.cm.gray)
    # ax.set_title('can')
    # ax.axis('off')
    # FILLING HOLD FROM EDGE DETECTION
    filled_img = ndi.binary_fill_holes(can)
    # fig, ax = plt.subplots(figsize=(4, 3))
    # ax.imshow(filled_img, cmap=plt.cm.gray)
    # ax.set_title('filled_image')
    # ax.axis('off')
    # DILATION FOR TEXT DETECTION
    dilationImage = closing(filled_img, rectangle(5, const.TOP_VIEW_DILATION_HORIZONTAL_SIZE))
    # fig, ax = plt.subplots(figsize=(4, 3))
    # ax.imshow(dilationImage, cmap=plt.cm.gray)
    # ax.set_title('bw')
    # ax.axis('off')
    # plt.show()
    interestedArea = CustomImageClass(minInterestHeight, maxInterestHeight, minInterestWidth, maxInterestWidth)
    return dilationImage, binaryImage, interestedArea


# SEGMENTAION AND GET CROPPED TEXT FROM IMAGE

def doGetCroppedTextFromImage(dilationImage, binaryImage, ax, interestedArea, outputFolderPath):
    labeledImage = label(dilationImage)
    for region in regionprops(labeledImage):
        # take regions with large enough areas
        if region.area >= const.MIN_REGION_AREA_GROUP_TEXT:
            # GET COORDINATE
            minr, minc, maxr, maxc = region.bbox
            # VALIDATE ASPECT RATIO
            if const.MIN_ASPECT_RATIO_GROUP_TEXT < (maxc - minc) / (maxr - minr) < const.MAX_ASPECT_RATIO_GROUP_TEXT:
                minCroppedX = interestedArea.getMinX()
                minCroppedY = interestedArea.getMinY()
                # ADD PADDING
                minr -= const.TEXT_IMAGE_CROP_PADDING_SIZE
                maxr += const.TEXT_IMAGE_CROP_PADDING_SIZE
                minc -= const.TEXT_IMAGE_CROP_PADDING_SIZE
                maxc += const.TEXT_IMAGE_CROP_PADDING_SIZE
                # SET DRAW RECTANGLE COORDINATE
                groupTextMinY = minCroppedY + minr
                groupTextMaxY = minCroppedY + maxr
                groupTextMinX = minCroppedX + minc
                groupTextMaxX = minCroppedX + maxc
                # SAVE CROPPED IMAGE
                cropped = binaryImage[minr:maxr, minc:maxc]
                # GET INNER TEXT
                labelText = label(cropped)
                regionsCroppedLabel = regionprops(labelText)
                if len(regionsCroppedLabel) >= const.MIN_NUMBER_OF_TEXT_IN_GROUP_TEXT:
                    # DRAW RECTANGLE GROUP TEXT
                    rect = mpatches.Rectangle((groupTextMinX, groupTextMinY), groupTextMaxX - groupTextMinX,
                                              groupTextMaxY - groupTextMinY, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    # SAVE GROUP TEXT
                    io.imsave(
                        outputFolderPath + const.REGION_GROUP_TEXT_FOLDER_NAME + 'groupText' + getDateTimeStrWithFormat(
                            const.DATE_TIME_PATTERN_FOLDER_NAME) + '.png', img_as_ubyte(cropped))
                    # INDEX FILE
                    indexFile = 1
                    for regionText in regionsCroppedLabel:
                        minrTxt, mincTxt, maxrTxt, maxcTxt = regionText.bbox
                        # VALIDATE RATIO TEXT
                        if const.MIN_ASPECT_RATIO_TEXT < (maxcTxt - mincTxt) / (
                                maxrTxt - minrTxt) < const.MAX_ASPECT_RATIO_TEXT:
                            # TEXT COORDINATE
                            minrS = minr + minrTxt - const.TEXT_IMAGE_CROP_PADDING_SIZE
                            maxrS = minr + maxrTxt + const.TEXT_IMAGE_CROP_PADDING_SIZE
                            mincS = minc + mincTxt - const.TEXT_IMAGE_CROP_PADDING_SIZE
                            maxcS = minc + maxcTxt + const.TEXT_IMAGE_CROP_PADDING_SIZE
                            textRecMinY = groupTextMinY + minrTxt - const.TEXT_IMAGE_CROP_PADDING_SIZE
                            textRecMaxY = groupTextMinY + maxrTxt + const.TEXT_IMAGE_CROP_PADDING_SIZE
                            textRecMinX = groupTextMinX + mincTxt - const.TEXT_IMAGE_CROP_PADDING_SIZE
                            textRecMaxX = groupTextMinX + maxcTxt + const.TEXT_IMAGE_CROP_PADDING_SIZE
                            # DRAW RECTANGLE
                            rect = mpatches.Rectangle((textRecMinX, textRecMinY), textRecMaxX - textRecMinX,
                                                      textRecMaxY - textRecMinY,
                                                      fill=False, edgecolor='blue', linewidth=2)
                            ax.add_patch(rect)
                            croppedText = binaryImage[minrS:maxrS, mincS:maxcS]
                            # viewr = ImageViewer(croppedText)
                            # viewr.show()
                            io.imsave(
                                outputFolderPath + const.REGION_SPLIT_TEXT_FOLDER_NAME + 'spltText' + str(
                                    indexFile) + '.png', img_as_ubyte(croppedText))
                            indexFile += 1


def getInterestCroppedArea(image):
    imageHeight, imageWidth = image.shape
    minHeightInput = int(imageHeight * (1 - const.TOP_VIEW_PROCESS_HORIZONTAL_PART))
    minWidthInput = int(imageWidth * (1 - const.TOP_VIEW_PROCESS_VERTICLE_PART) / 2)
    maxWidthInput = int(imageWidth - minWidthInput)
    return minHeightInput, imageHeight, minWidthInput, maxWidthInput


def isRegionInInterestingArea(region, interestedArea):
    minr, minc, maxr, maxc = region.bbox
    # VALIDATE WIDTH
    if interestedArea.getMinY() <= minr <= interestedArea.getMaxY() and interestedArea.getMinY() <= maxr <= interestedArea.getMaxY():
        if interestedArea.getMinX() <= minc <= interestedArea.getMaxX() and interestedArea.getMinX() <= maxc <= interestedArea.getMaxX():
            # PASS
            return True
    # FAIL
    return False


def createFolderOutputByTime():
    # CREATE FOLDER FOR OUTPUT PATH
    outputPath = '../' + const.OUTPUT_FOLDER_NAME
    try:
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
    except OSError:
        print("Creation of the directory %s failed" % outputPath)

    # CREATE FOLDER FOR OUTPUT IMAGE
    outputFolderPath = outputPath + getDateTimeStrWithFormat(
        const.DATE_TIME_PATTERN_FOLDER_NAME) + '/'
    try:
        os.mkdir(outputFolderPath)
    except OSError:
        print("Creation of the directory %s failed" % outputFolderPath)

    # CREATE FOLDER INPUT IMAGE
    inputImageFolderPath = outputFolderPath + const.INPUT_FOLDER_NAME
    try:
        os.mkdir(inputImageFolderPath)
    except OSError:
        print("Creation of the directory %s failed" % inputImageFolderPath)

    # CREATE FOLDER OUTPUT RECTANGLE IMAGE
    outputRecImageFolderPath = outputFolderPath + const.RECTANGLE_IMAGE_FOLDER_NAME
    try:
        os.mkdir(outputRecImageFolderPath)
    except OSError:
        print("Creation of the directory %s failed" % outputRecImageFolderPath)

    # CREATE FOLDER OUTPUT GROUP TEXT
    outputGroupTextFolderPath = outputFolderPath + const.REGION_GROUP_TEXT_FOLDER_NAME
    try:
        os.mkdir(outputGroupTextFolderPath)
    except OSError:
        print("Creation of the directory %s failed" % outputGroupTextFolderPath)

    # CREATE FOLDER OUTPUT TEXT
    outputTextFolderPath = outputFolderPath + const.REGION_SPLIT_TEXT_FOLDER_NAME
    try:
        os.mkdir(outputTextFolderPath)
    except OSError:
        print("Creation of the directory %s failed" % outputTextFolderPath)

    return outputFolderPath


def getDateTimeStrWithFormat(format):
    return datetime.now().strftime(format)
