import skimage.filters as filter
import skimage.feature as fea
from skimage import io
from skimage.color import rgb2gray
from skimage.viewer import ImageViewer
from skimage import morphology
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import matplotlib.patches as mpatches
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
from skimage import draw
import numpy as np

# PREPARE INPUT
image = io.imread('../images/Top/thumbnail_IMG_9517.jpg', True)
denoise_image = denoise_tv_chambolle(image)
thresh = filter.threshold_yen(denoise_image)
binary = image > thresh
can = fea.canny(binary)
filled_img = ndi.binary_fill_holes(can)
# REMOVE SMALL OBJECT AFTER CANNY
removeSmallPic = morphology.remove_small_objects(filled_img, 180)
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(removeSmallPic, cmap=plt.cm.gray)
ax.set_title('removeSmallPic')
ax.axis('off')
#### PROCESS
bw = morphology.closing(removeSmallPic, morphology.rectangle(5,100))
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(bw, cmap=plt.cm.gray)
ax.set_title('closing')
ax.axis('off')

## LABEL
con = label(bw)
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
for region in regionprops(con):
    # take regions with large enough areas
    if region.area >= 200:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc-5, minr-5), maxc - minc + 10, maxr - minr + 10,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        # crop & save image
        # viewere = ImageViewer(region.image)
        # viewere.show()
        # io.imsave('test.jpg', img_as_ubyte(region.image))

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
