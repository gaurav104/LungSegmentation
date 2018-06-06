import numpy as np
import os
from skimage import transform, io, img_as_float, exposure

def loadDataGeneral():
    """Function for loading arbitrary data in standard formats"""
    path = 'CXR_png/'
    X, y = [], []
    for i, filename in enumerate(os.listdir(path)):
        img = img_as_float(io.imread(path +filename[:-4] + '.png'))
        mask = io.imread('Mask/' +filename[:-4]+'msk.png')
        #img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        #mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print ('### Dataset loaded')
    print ('\t{}'.format(path))
    print ('\t{}\t{}'.format(X.shape, y.shape))
    print ('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y