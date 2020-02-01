import skimage.filters as filter
from skimage import io
from skimage.color import rgb2gray
from skimage.viewer import ImageViewer
from skimage import morphology
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import matplotlib.patches as mpatches

image = io.imread('../images/Top/thumbnail_IMG_9517.jpg', True)
thresh = filter.threshold_yen(image)
binary = image > thresh
dila = morphology.dilation(binary, morphology.disk(3))
con = label(dila)
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
print(con)
for region in regionprops(con):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        # crop & save image
        viewere = ImageViewer(region.image)
        viewere.show()
        io.imsave('test.jpg', img_as_ubyte(region.image))

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
