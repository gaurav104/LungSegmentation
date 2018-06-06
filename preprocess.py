import os
import numpy as np
from skimage import io, exposure
from __future__ import print_function


def make_masks():
    path = 'CXR_png/'
    for i, filename in enumerate(os.listdir(path)):
        left = io.imread('ManualMask/leftMask/' + filename[:-4] + '.png')
        right = io.imread('ManualMask/rightMask/' + filename[:-4] + '.png')
        io.imsave('Mask/' + filename[:-4] + '.png', np.clip(left + right, 0, 255))
        print ('Mask', i, filename)

make_masks()
    