
# coding: utf-8
import os, sys
from PIL import Image
import numpy as np
from matplotlib import image as mpimg
from matplotlib import pyplot as plt



# convert all image to other type under current path
def img_all_convert(current_path, current_type, new_type, new_path):
    '''
    parameters:
    current_path -- current images's directory
    current_type -- current images's type
    new_type -- convert image into target type
    new_path -- convert image to target directory. 
    '''
    dirs = os.listdir(current_path)
    for infile in dirs:
        f, e = os.path.splitext(infile)
        if e == current_type:
            outfile = f + new_type
            with Image.open(current_path + '/' + infile) as image:
                image.save(new_path + '/' + outfile)


#current_path = '/Users/nataliezhu/Documents/sem_computer_vision/images'
#new_path = '/Users/nataliezhu/Documents/sem_computer_vision/sample'
#img_all_convert(current_path, '.tif', '.jpg', new_path)


# resize all images under the current path
def img_all_resize(current_path, new_path, new_size):
    '''
    parameters:
    current_path -- current images's directory
    new_path -- convert image to target directory.
    new_h -- convert image into new (height, weight)
    '''
    dirs = os.listdir(current_path)
    for infile in dirs:
        if not infile.startswith('.'):
            with Image.open(current_path + '/' + infile) as image:
                new_image = image.resize(new_size, Image.ANTIALIAS)
                new_image.save(new_path + '/' + infile)


# r_current_path = '/Users/nataliezhu/Documents/sem_computer_vision/img/nw'
# r_new_path = '/Users/nataliezhu/Documents/sem_computer_vision/sample'
# img_resize(r_current_path, r_new_path, (1000,1000))


# convert all RGB images to greyscale under current path
def img_all_to_grey(current_path, new_path):
    '''
    parameters:
    current_path -- current images's directory
    new_path -- convert image to target directory. 
    '''
    dirs = os.listdir(current_path)
    for infile in dirs:
        if not infile.startswith('.'):
            with Image.open(current_path + '/' + infile) as image:
                new_image = image.convert('L')
                new_image.save(new_path + '/' + infile)


# g_current_path = '/Users/nataliezhu/Documents/sem_computer_vision/img/nw'
# g_new_path = '/Users/nataliezhu/Documents/sem_computer_vision/sample'
# img_to_grey(g_current_path, g_new_path)
# 
# img = mpimg.imread('/Users/nataliezhu/Documents/sem_computer_vision/sample/GaAsP NW.jpg')
# 
# img.shape
# 
# plt.imshow(img)
# plt.show()
