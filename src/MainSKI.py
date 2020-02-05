import skimage.filters as filter
import skimage.feature as fea
from skimage import io
from skimage.util import crop
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
import time;
from skimage import img_as_int

# PREPARE INPUT
image = io.imread('../images/Top/thumbnail_IMG_9517.jpg', True)
denoise_image = denoise_tv_chambolle(image)
thresh = filter.threshold_yen(denoise_image)
binary = image > thresh
can = fea.canny(binary)
filled_img = ndi.binary_fill_holes(can)
# REMOVE SMALL OBJECT AFTER CANNY
removeSmallPic = morphology.remove_small_objects(filled_img, 180)
#### PROCESS
bw = morphology.closing(removeSmallPic, morphology.rectangle(5,100))
## LABEL
con = label(bw)
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
for region in regionprops(con):
    # take regions with large enough areas
    if region.area >= 200:
        # GET COORDINATE
        minr, minc, maxr, maxc = region.bbox
        # VALIDATE ASPECT RATIO
        if 7 < (maxc-minc) / (maxr-minr) < 10:
            # TODO GET INNER TEXT
            for regionText in regionprops(img_as_int(region.image)):
                minrTxt, mincTxt, maxrTxt, maxcTxt = regionText.bbox
                # DRAW RECTANGLE
                rect = mpatches.Rectangle((mincTxt - 5, minrTxt - 5), maxcTxt - mincTxt + 10, maxrTxt - minrTxt + 10,
                                          fill=False, edgecolor='blue', linewidth=2)
                ax.add_patch(rect)
            # DRAW RECTANGLE
            rect = mpatches.Rectangle((minc-5, minr-5), maxc - minc + 10, maxr - minr + 10,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            # SAVE CROPPED IMAGE
            cropped = binary[minr:maxr, minc:maxc]
            io.imsave('../output/region-image'+str(time.time())+'.png', img_as_ubyte(cropped))
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('../output/full-image'+str(time.time())+'.png')
plt.show()

