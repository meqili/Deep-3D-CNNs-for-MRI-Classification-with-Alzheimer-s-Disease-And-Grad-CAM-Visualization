#!/usr/bin/env python
# coding: utf-8

#This bash script is for 3D-CNN based Gradient-weighted Class Activation Mapping method (3D-GradCAM)
#author:Qi Li
#github:liqi814
##Without Singularity
##python3.5 resnet_3d_grad_cam.py img_path cam_npz_path mri_npz_path prefix
##With Singularity
##singularity exec --nv classify.img python3.5 resnet_3d_grad_cam.py imgpath cam_npz_path mri_npz_path prefix


# Import libraries
import sys
import time
#from nilearn.image import resample_img
#import pylab as plt
import nibabel as nb
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.transform import rotate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


cam =np.load(sys.argv[1])
mri = np.load(sys.argv[2])
prefix = sys.argv[3]

cam=np.array(cam['arr_0'])
#print(cam)
cam = resize(cam,(110,110,110))

#print(cam)
mri=np.array(mri['arr_0'])


heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
#print("heatmap size", heatmap.shape)
#print(heatmap)

def choose_slice(mri_slice, heatmap_slice):
  mri_slice = rotate(mri_slice,90)
  heatmap_slice = rotate(heatmap_slice,90)
  image = np.float32(mri_slice).astype('bool') * np.float32(heatmap_slice)
  #print(image.max())
  return image
  

fig = plt.figure(figsize=(25, 25))
for i in range(110):
  plt.subplot(11, 10, i+1)
  #image = choose_slice(mri[:, i, :], heatmap[:, i, :])
  image = choose_slice(mri[i, :, :], heatmap[i, :, :])
  #image = choose_slice(mri[:, :, i], heatmap[:, :, i])
  plt.imshow(image, interpolation=None, vmax=1., vmin=.0, alpha=1,  cmap=plt.cm.rainbow)

cbar = plt.colorbar()
#plt.show()

pic_name = 'pics/' + prefix + '_all_img.png'
plt.savefig(pic_name)

#plt.imshow(np.float32(cov_heatmap))
fig = plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
image = choose_slice(mri[:, 50, :], heatmap[: , 50, :])
plt.imshow(image, interpolation=None, vmax=1, vmin=.0, alpha=1,  cmap=plt.cm.rainbow)

plt.subplot(2,3,2)
image = choose_slice(mri[50, :, :], heatmap[50 , :, :])
plt.imshow(image, interpolation=None, vmax=1, vmin=.0, alpha=1,  cmap=plt.cm.rainbow)
#plt.imshow(rotate(cam[50 , :, :],90),interpolation=None, vmax=1, vmin=.0, alpha=1,  cmap=plt.cm.rainbow)

plt.subplot(2, 3, 3)
image = choose_slice(mri[:, :, 50], heatmap[: , :, 50])
plt.imshow(image, interpolation=None, vmax=1, vmin=.0, alpha=1,  cmap=plt.cm.rainbow)
cbar = plt.colorbar()

plt.subplot(2, 3, 4)
#plt.imshow(rotate(mri[:, 50, :],90), interpolation=None, vmax=1, vmin=.0, alpha=1,  cmap=plt.cm.rainbow)
plt.imshow(rotate(mri[:, 50, :],90), cmap=plt.cm.Greys_r)
#plt.imshow(matr, cmap=plt.cm.Greys_r, interpolation=None, vmax=1., vmin=0.)
#plt.hold(True)
plt.subplot(2,3,5)
plt.imshow(rotate(mri[50, :, :],90), cmap=plt.cm.Greys_r)
#plt.imshow(rotate(mri[50, :, :],90), interpolation=None, vmax=1, vmin=.0, alpha=1,  cmap=plt.cm.rainbow)
#cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
plt.subplot(2, 3, 6)
#plt.imshow(rotate(mri[:, :, 50],90), interpolation=None, vmax=1, vmin=.0, alpha=1,  cmap=plt.cm.rainbow)
plt.imshow(rotate(mri[:, :, 50],90), cmap=plt.cm.Greys_r)
cbar = plt.colorbar()
#plt.show()
pic_name = 'pics/' + prefix + '_img.png'
plt.savefig(pic_name)
#plt.savefig('pics/S144140_vgg.png')
