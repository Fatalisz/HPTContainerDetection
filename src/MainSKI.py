import matplotlib.pyplot as plt
from ContainerDetectionImageUtils import preProcessImage, doGetCroppedTextFromImage
from skimage import io
import time

if __name__ == '__main__':
    # PREPARE INPUT
    image = io.imread('../images/ContainerAllSides/Top/20200127184934380T.jpg', True)
    #image = io.imread('../images/Top/thumbnail_IMG_9517.jpg', True)
    imageForProcess, binaryImage = preProcessImage(image)
    # INIT PY_PLOT
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    doGetCroppedTextFromImage(imageForProcess, binaryImage, ax)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('../output/full-image/full-image'+str(time.time())+'.png')
    plt.show()


