import numpy as np

from skimage import filters
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.morphology import closing, square
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import clear_border

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

import shutil
import os
import scipy.misc

LABEL_FILE = './my_label.txt'
GRAPH_FILE = './my_graph.pb'

ECCENTRICYTY = 0.995
SOLIDITY = 0.3
EXTENT = 0.2
EULER = -4
RATIO = 3


class textDetection:
    def __init__(self, image_path):
        self.image = imread('../images/Top/thumbnail_IMG_9517.jpg', True)
        self.bw = self.imagePreProcess()

    def imagePreProcess(self):
        image = denoise_tv_chambolle(self.image, weight=0.1)
        thres = filters.threshold_yen(image)
        bw = closing(image > thres, square(2))
        clear_border(bw)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(bw, cmap=plt.cm.nipy_spectral)
        ax.set_title('markers')
        ax.axis('off')
        return bw

    def getTextCandidate(self):
        label_black = label(self.bw, background=1)
        label_white = label(self.bw, background=0)
        candidateValue = []
        candidatePosition = []

        for region in regionprops(label_black):
            minr, minc, maxr, maxc = region.bbox
            if region.eccentricity > ECCENTRICYTY:
                continue
            if region.solidity < SOLIDITY:
                continue
            if region.extent < EXTENT:
                continue
            if region.euler_number < EULER:
                continue
            if (maxc - minc) / (maxr - minr) > RATIO:
                continue
            if region.area > 10:
                margin = 3
                minr = minr - margin
                minc = minc - margin
                maxr = maxr + margin
                maxc = maxc + margin
                sample = self.bw[minr:maxr, minc:maxc]
                if sample.shape[0] * sample.shape[1] == 0:
                    continue
                else:
                    # sample = resize(sample, (50, int(sample.shape[1] * (50 / sample.shape[0]))))
                    candidateValue.append(sample)
                    candidatePosition.append(region.bbox)

        for region in regionprops(label_white):
            minr, minc, maxr, maxc = region.bbox
            if region.eccentricity > ECCENTRICYTY:
                continue
            if region.solidity < SOLIDITY:
                continue
            if region.extent < EXTENT:
                continue
            if region.euler_number < EULER:
                continue
            if (maxc - minc) / (maxr - minr) > RATIO:
                continue
            if region.area > 10:
                margin = 3
                minr = minr - margin
                minc = minc - margin
                maxr = maxr + margin
                maxc = maxc + margin
                sample = self.bw[minr:maxr, minc:maxc]
                if sample.shape[0] * sample.shape[1] == 0:
                    continue
                else:
                    # sample = resize(sample, (50, int(sample.shape[1] * (50 / sample.shape[0]))))
                    sample = 1 - sample
                    candidateValue.append(sample)
                    candidatePosition.append(region.bbox)

        num = np.array(candidatePosition).shape[0]
        need_to_del = False
        delete = []
        for i in range(num):
            minr1 = candidatePosition[i][0]
            minc1 = candidatePosition[i][1]
            maxr1 = candidatePosition[i][2]
            maxc1 = candidatePosition[i][3]
            for j in range(num):
                minr2 = candidatePosition[j][0]
                minc2 = candidatePosition[j][1]
                maxr2 = candidatePosition[j][2]
                maxc2 = candidatePosition[j][3]
                if(minc1 > minc2):
                    if(minr1 > minr2):
                        if(maxc1 < maxc2):
                            if(maxr1 < maxr2):
                                need_to_del = True
            if need_to_del is True:
                delete.append(i)
            need_to_del = False
        for i in range(np.array(delete).shape[0]):
            del candidateValue[delete[i] - i]
            del candidatePosition[delete[i] - i]
        self.candidate = {'fullscale': np.array(candidateValue),
                          'position': np.array(candidatePosition)
                          }
        num = self.candidate['position'].shape[0]
        print('Trich xuat duoc: ', num, ' mau')

    def showCandidate(self):
        if self.candidate['position'].shape[0] == 0:
            print('Chua co candidate')
            print('Su dung ham getTextCandidate()')
            return
        fig, ax = plt.subplots(1)
        ax.imshow(self.image)
        for i in range(self.candidate['position'].shape[0]):
            minr = self.candidate['position'][i][0]
            minc = self.candidate['position'][i][1]
            maxr = self.candidate['position'][i][2]
            maxc = self.candidate['position'][i][3]
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.set_axis_off()
        plt.show()