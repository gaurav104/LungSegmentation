

import image_slicer
import os
from tqdm import tqdm
path  = 'Mask/Train/Images/'
img_names = os.listdir(path)

for i in tqdm(img_names):
    tiles = image_slicer.slice(path+i,4, save=False)
    image_slicer.save_tiles(tiles, directory='Split_Mask/Train/Images',prefix = i[:-4])