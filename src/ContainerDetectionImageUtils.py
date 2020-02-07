from skimage import filters, io, img_as_ubyte
from skimage.feature import canny
from skimage.morphology import remove_small_objects, closing, rectangle
from skimage.measure import label, regionprops
from skimage.viewer import ImageViewer
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import try_all_threshold
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage as ndi
import time
import ContainerDetectionConstant as const

# PREPARE IMAGE FOR SEGMENTATION
def preProcessImage(image):
    # DENOISE IMAGE
    denoiseImage = denoise_tv_chambolle(image)
    # CROP INTEREST PART FOR THRESHOLD
    imageHeight, imageWidth = image.shape
    minHeightInput = int(imageHeight*(1-const.TOP_VIEW_PROCESS_HORIZONTAL_PART))
    minWidthInput = int(imageWidth*(1-const.TOP_VIEW_PROCESS_VERTICLE_PART)/2)
    maxWidthInput = int(imageWidth - minWidthInput)
    croppedInputImage = image[minHeightInput:imageHeight, minWidthInput:maxWidthInput]
    # fig, ax = plt.subplots(figsize=(4, 3))
    # ax.imshow(croppedInputImage, cmap=plt.cm.gray)
    # ax.set_title('input')
    # ax.axis('off')
    # TEST THRESHOLD
    # fig, ax = try_all_threshold(croppedInputImage, figsize=(10, 8), verbose=False)
    # plt.show()
    # GET THRESHOLD IMAGE
    thresh = filters.threshold_yen(croppedInputImage)
    binaryImage = image > thresh
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
    # REMOVE SMALL OBJECT IN IMAGE
    removeSmallPic = remove_small_objects(filled_img, const.TOP_VIEW_REMOVE_SMALL_PIC_MIN_SIZE)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(removeSmallPic, cmap=plt.cm.gray)
    ax.set_title('removeSmallPic')
    ax.axis('off')
    # DILATION FOR TEXT DETECTION
    dilationImage = closing(removeSmallPic, rectangle(5, const.TOP_VIEW_DILATION_HORIZONTAL_SIZE))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(dilationImage, cmap=plt.cm.gray)
    ax.set_title('bw')
    ax.axis('off')
    plt.show()
    return dilationImage, binaryImage

# SEGMENTAION AND GET CROPPED TEXT FROM IMAGE

def doGetCroppedTextFromImage(dilationImage, binaryImage, ax):
    labeledImage = label(dilationImage)
    for region in regionprops(labeledImage):
        # take regions with large enough areas
        if region.area >= const.MIN_REGION_AREA:
            # GET COORDINATE
            minr, minc, maxr, maxc = region.bbox
            # VALIDATE ASPECT RATIO
            if const.MIN_ASPECT_RATIO_GROUP_TEXT < (maxc - minc) / (maxr - minr) < const.MAX_ASPECT_RATIO_GROUP_TEXT:
                # DRAW RECTANGLE
                rect = mpatches.Rectangle((minc - 5, minr - 5), maxc - minc + 10, maxr - minr + 10,
                                          fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                # SAVE CROPPED IMAGE
                cropped = binaryImage[minr:maxr, minc:maxc]
                # GET INNER TEXT
                removeSmallPicText = remove_small_objects(cropped, 50)
                labelText = label(removeSmallPicText)
                regionsCroppedLabel = regionprops(labelText)
                if len(regionsCroppedLabel) >= const.MIN_NUMBER_OF_TEXT_IN_GROUP_TEXT:
                    io.imsave('../output/region-group-text/groupText' + str(time.time()) + '.png', img_as_ubyte(cropped))
                    for regionText in regionsCroppedLabel:
                        minrTxt, mincTxt, maxrTxt, maxcTxt = regionText.bbox
                        # VALIDATE RATIO TEXT
                        minrS = minr + minrTxt
                        maxrS = minr + maxrTxt
                        mincS = minc + mincTxt
                        maxcS = minc + maxcTxt
                        # DRAW RECTANGLE
                        rect = mpatches.Rectangle((mincS - 5, minrS - 5), maxcS - mincS + 10, maxrS - minrS + 10,
                                                  fill=False, edgecolor='blue', linewidth=2)
                        ax.add_patch(rect)
                        croppedText = binaryImage[minrS:maxrS, mincS:maxcS]
                        io.imsave('../output/region-split-text/spltText' + str(time.time()) + '.png', img_as_ubyte(croppedText))
                        viewr = ImageViewer(regionText.image)
                        viewr.show()