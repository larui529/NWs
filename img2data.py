
# coding: utf-8

# This is a list of function to process nw images and not nw images into train and test dataset

# In[11]:


import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from IPython.display import display, Image
get_ipython().magic(u'matplotlib inline')
from numpy import random
import shutil


# In[3]:


# extract function to extract all the subfolder under root
def extract(root):
    """
    Argument: 
    root -- name or dir of a folder
    Return:
    data_folder -- subfolders of root
    """
    data_folders = [(os.path.join(root, d),d) for d in sorted(os.listdir(root)) if not d.startswith('.')]
    #print data_folders
    return data_folders


# In[4]:


def file_split(filenames, train_size):
    """
    Argument:
    filenames -- list of filenames
    train_size  -- ratio of train dataset (e.g. 0.8)
    Return:
    train_filenames -- list of train filenames
    test_filenames -- list of test filenames
    """
    filenames = sorted(filenames)
    random.seed(230)
    random.shuffle(filenames)
    split_1 = int(train_size * len(filenames))
    train_filenames = filenames[:split_1]
    test_filenames = filenames[split_1:]
    return train_filenames, test_filenames
    


# In[5]:


# load function to load imgs from folder (contains several subfolders) and output a dataset and a labels
# import image


def load(data_folders,max_num_images, image_height, image_width, pixel_depth=255):
    """
    Argument: 
    data_folders -- folders of different classes. Currently is nw and not_nw
    max_num_images -- max number of images can pad into dataset. 
    image_height -- height of image
    image_width -- width of image
    pixel_depth -- depth of image, usually 225.0
    Return:
    Dataset -- dataset contains all the images in the folder, dimesion is [num_images, image_height, image_weith]
    labels -- labels of images, dimension in [num_images], data range in range(classes)
    """
    #print data_folders
    category_dict = {}
    dataset = np.ndarray(  # create a ndarray with (max_num_images, n_H, n_W)
        shape=(max_num_images, image_height, image_width), dtype=np.float32) 
    labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
    label_index = 0 # label index start from 0
    image_index = 0 # image_index start from 0
    for folder in data_folders:
        #print folder
        category_dict[folder[1]] = label_index
        #print os.listdir(folder)
        for image in os.listdir(folder[0]):
            image_file = os.path.join(folder[0], image)
            try:
                image_data = (plt.imread(image_file).astype(float) -
                              pixel_depth / 2) / pixel_depth # normalize image
                if image_data.shape != (image_height, image_width):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[image_index, :, :] = image_data
                labels[image_index] = label_index
                image_index += 1 # read another image
            except IOError as e:
                print 'Could not read:', image_file, ':', e, '- it\'s ok, skipping.'
        label_index += 1 # next folder, next label
    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    labels = labels[0:num_images]
    print 'Full dataset tensor:', dataset.shape
    print 'Mean:', np.mean(dataset)
    print 'Standard deviation:', np.std(dataset)
    print 'Labels:', labels.shape
    return dataset, labels, category_dict


# In[9]:


def train_test_split_folder(nw_folder_path, not_nw_folder_path, train_folders_path, test_folders_path, train_size):
    """
    Argument:
    nw_folder_path -- path to the nw imgs folder
    not_nw_folder_path -- path to the not_nw imgs folder
    train_folder_path -- folder of train dataset, will contain 2 sub dataset
    test_folder_path -- folder of test dataset, will contain 2 sub dataset
    Return:
    None -- just need to copy and split pictures in nw_folder, not_nw_folder and put them in train_folder and test_folder
    """
    if not os.path.exists(train_folders_path) : # create train_folder if not exist, also create sub folder
        os.makedirs(train_folders_path)
        os.makedirs(train_folders_path + "/nws")
        os.makedirs(train_folders_path + "/not_nws")
    if not os.path.exists(test_folders_path): # create test_folder if not exist, also create sub folder
        os.makedirs(test_folders_path)
        os.makedirs(test_folders_path + "/nws")
        os.makedirs(test_folders_path + "/not_nws")
    nw_filenames = os.listdir(nw_folder_path) # list of all the nw imgs 
    not_nw_filenames = os.listdir(not_nw_folder_path) # list of all the non_nw imgs
    
    nw_train, nw_test = file_split(nw_filenames, train_size) # split nw imgs and output filenames
    not_nw_train, not_nw_test = file_split(not_nw_filenames, train_size) # split not_nw imgs and output filenames
    
    imgs = (nw_train, nw_test, not_nw_train, not_nw_test)
    folder_names = (nw_folder_path, nw_folder_path, not_nw_folder_path, not_nw_folder_path)
    dest_names = (train_folders_path + "nws", test_folders_path + "nws",
                  train_folders_path + "not_nws", test_folders_path + "not_nws")
    print imgs
    print folder_names
    print dest_names

    for file_names, folder, dest in zip(imgs, folder_names, dest_names ):
        for file_name in file_names:
            full_file_name = os.path.join(folder, file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, dest)

