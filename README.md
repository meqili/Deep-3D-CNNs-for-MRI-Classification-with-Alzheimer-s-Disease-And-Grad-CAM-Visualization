# Deep 3D CNNs for MRI Classification with Alzheimer's Disease And Grad-CAM for Visualization

This implemented paper is based on the work [Residual and Plain Convolutional Neural Networks for 3D Brain MRI Classification](https://arxiv.org/abs/1701.06643) and its implementation in this [repo](https://github.com/neuro-ml/resnet_cnn_mri_adni). Please cite properly if you are using code in this repo.

## Singularity or Requirment

This repo has the receipts for buiding singularity containers, or you can setup the environment by yourself based on the receipts. Please check the folder [singularity_receipt](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/singularity_receipt).
```bash
#buiding singularity containers
sudo singularity build resnet_cnn_mri.def classify.img
sudo singularity build cam_vis.img cam_vis.def
```
## 1. MRI Classification

### 1.1 Data Preparation (folder [data](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/data))

| File Name | Description |
| ------------- | ------------- |
| all_metadata.csv  | All image data |
| metadata.csv  | Training dataset  |
| test.csv  | Test dataset  |

### 1.2 Classification

```bash
module load singularity

##Use the VGG network:
nohup singularity exec --nv classify.img python3.5 scripts/vgg_3d_pred.py > vgg.out &

##Use the Resnet network:
nohup singularity exec --nv classify.img python3.5 scripts/res_3d_pred.py >resnet.out &
```

### 1.3 Result
 
Please check the folders [results_vgg](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/results_vgg) and [results_resnet](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/results_resnet) for more details.


## 2. Grad-CAM Visualization

### 1.1 Grad-CAM
After the classification, the models have learned the weights from images. Choose one image in which you would love to view the discriminative regions. In this example, I used *S117504-reg.nii.gz*. And also, you can change to the layer that you want to visualize.

Usage:
```bash
##Without Singularity
python3.5  python_script  imgpath prefix
##With Singularity
Singularity  exec --nv classify.img  python3.5 python_script  imgpath  prefix
```

For example:
```bash
qli@gpu001$ singularity exec --nv classify.img python3.5 scripts/vgg_3d_grad_cam.py /home/qli/AlzheimerClassify/5.Resize/S117504-reg.nii.gz S117504
Using gpu device 0: Tesla P100-PCIE-12GB (CNMeM is disabled, cuDNN 5110)
[[ 0.00110552  0.99889451]]
test_loss 0.001106127048842609
-----------
~/mri_classif


qli@gpu001$ singularity exec --nv classify.img python3.5 scripts/resnet_3d_grad_cam.py /home/qli/AlzheimerClassify/5.Resize/S117504-reg.nii.gz S117504
Using gpu device 0: Tesla P100-PCIE-12GB (CNMeM is disabled, cuDNN 5110)
[[ 0.12858798  0.87141204]]
test_loss 2.0511419773101807
-----------
~/mri_classif
```

From the examples, the scripts will return some info below:
```bash
[[ 0.12858798  0.87141204]]
test_loss 2.0511419773101807
```
The two number inside the double square brackets stands for the possibilities of this image is classified as normal and Alzheimer’s respectively.

### 1.2 Debugging

If you got a problem like below:
```bash
_tkinter.TclError: no display name and no $DISPLAY environment variable
```

Here is the solution:
```bash
echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
```


### 1.3 Visualization
After running the Grad-CAM, two *.npz* files (one contains the discriminative info and another one contains the MRI data) will be saved in folder [npz_res](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/npz_res) or [npz_vgg](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/npz_vgg). Scripts in this section do not need to run on the GPU cards. 

```bash
#Usage:
singularity exec cam_vis.img python3.6 scripts/visualize_cam.py cam_npz/file/path mri_npz/file/path prefix

#Examples:

singularity exec cam_vis.img python3.6 scripts/visualize_cam.py npz_vgg/S117504_cam_conv4c.npz npz_vgg/S117504_mri.npz S117504_vgg

singularity exec cam_vis.img python3.6 scripts/visualize_cam.py npz_res/S117504_cam_voxres9_conv2.npz npz_res/S117504_mri.npz S117504_res
```

