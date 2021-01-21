# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:44:43 2020

@author: ethan
"""
import matplotlib.pyplot as plt
import numpy as np
import os 
from libtiff import TIFF

def plot_side_by_side(im1,im2):
  f = plt.figure()
  f.add_subplot(1,2, 1)
  plt.imshow(np.rot90(im1,2))
  f.add_subplot(1,2, 2)
  plt.imshow(np.rot90(im2,2))
  plt.show(block=True)
  
  
  
def show_me_img(path):#show a single image
  tif=TIFF.open(path)
  image=tif.read_image()
  plt.imshow(image,interpolation='nearest')
  plt.show()


#This function shows evry image in the folder , sorted as image and mask
def show_me_folder(path):
  for file in sorted(os.listdir(path)):
    #if file.endswith(change):#on colab, deleting files will add some weird stuff on the folder
    base_direc=path+'/'
    print(file)
    tif=TIFF.open(base_direc+file)
    image=tif.read_image()
    plt.imshow(image,interpolation='nearest')
    plt.show()

#Visualisation of albumentations transform

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
