import matplotlib.pyplot as plt
import ContainerDetectionImageUtils as imUtils
import ContainerDetectionConstant as const
from skimage import io, img_as_ubyte
import datetime
import numpy as np


def readImageToGrayScale(image):
    return io.imread(image, as_gray=True)


if __name__ == '__main__':
    # READ IMAGE
    imageCol = io.ImageCollection(const.INPUT_FOLDER_CONDITION)
    for image in imageCol:
        # GET FOLDER OUTPUT PATH
        outputFolderPath = imUtils.createFolderOutputByTime()
        # SAVE INPUT IMAGE
        io.imsave(outputFolderPath + const.INPUT_FOLDER_NAME + 'input' + imUtils.getDateTimeStrWithFormat(
            const.DATE_TIME_PATTERN_FOLDER_NAME) + '.png', img_as_ubyte(image))
        # PREPARE INPUT
        imageForProcess, binaryImage, interestedArea = imUtils.preProcessImage(image)
        # INIT PY_PLOT
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.gray)
        # DO SEGMENTATION AND CROP IMAGE
        imUtils.doGetCroppedTextFromImage(imageForProcess, binaryImage, ax, interestedArea, outputFolderPath)
        # SAVE RECTANGLE IMAGE
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(
            outputFolderPath + const.RECTANGLE_IMAGE_FOLDER_NAME + 'rec-image' + imUtils.getDateTimeStrWithFormat(
                const.DATE_TIME_PATTERN_FOLDER_NAME) + '.png')
        plt.close(fig)
        plt.cla()
        plt.clf()
        # plt.show()
