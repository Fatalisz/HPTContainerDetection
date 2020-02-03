import skimage.filters as filter
from skimage import io
from skimage.color import rgb2gray
from skimage.viewer import ImageViewer
from skimage import morphology
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import matplotlib.patches as mpatches
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import numpy as np

image = io.imread('../images/Top/thumbnail_IMG_9517.jpg', True)
denoise_image = denoise_tv_chambolle(image)
thresh = filter.threshold_yen(denoise_image)
binary = image > thresh
fill_coins = ndi.binary_fill_holes(binary)
removeSmallPic = morphology.remove_small_objects(fill_coins, 100)
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(removeSmallPic, cmap=plt.cm.gray)
ax.set_title('binary')
ax.axis('off')
bw = closing(removeSmallPic, morphology.disk(11))
clear_border(bw)
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(bw, cmap=plt.cm.gray)
ax.set_title('bw')
ax.axis('off')
dila = morphology.dilation(bw, morphology.disk(11))
con = label(dila)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(dila, cmap=plt.cm.gray)
ax.set_title('dilation')
ax.axis('off')

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
for region in regionprops(con):
    # take regions with large enough areas
    if region.area >= 200:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
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
