import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
from imageio import imread
from scipy.misc import imresize
import hickle as hkl
from kitti_settings import *

desired_im_sz = (128, 160)

def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im

im_list = []
source_list = []
folder = "validation"

im_dir = 'C:\\Users\\Mingtai\\Desktop\\prednet\\raw\\validation\\'
files = list(os.walk(im_dir, topdown=False))[-1][-1]
im_list += [im_dir + f for f in sorted(files)]
source_list += [folder] * len(files)

split = 'val'

print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
for i, im_file in enumerate(im_list):
    im = imread(im_file)
    X[i] = process_im(im, desired_im_sz)

hkl.dump(X, os.path.join(im_dir, 'X_' + split + '.hkl'))
hkl.dump(source_list, os.path.join(im_dir, 'sources_' + split + '.hkl'))
