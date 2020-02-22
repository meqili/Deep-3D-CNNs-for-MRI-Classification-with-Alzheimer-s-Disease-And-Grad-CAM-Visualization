#!/usr/bin/env python
# coding: utf-8

#This bash script is for 3D-CNN based Gradient-weighted Class Activation Mapping method (3D-GradCAM)
#author:Qi Li
#github:liqi814
##Without Singularity
##python3.5 vgg_3d_grad_cam.py imgpath prefix
##With Singularity
##singularity exec --nv classify.img python3.5 vgg_3d_grad_cam.py imgpath prefix


# Import libraries
import sys
import numpy as np
import nibabel as nib

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers.dnn import Conv3DDNNLayer
from lasagne.layers.dnn import Pool3DDNNLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import identity
from lasagne.layers import DropoutLayer


# Read data and load model
input_var = T.tensor5(name='input', dtype='float32')
target_var = T.ivector()
inp_shape = (None, 1, 110, 110, 110)
modelname = 'models/vgg_net_weights.npz'
imgpath = sys.argv[1]
prefix = sys.argv[2]
mx = nib.load(imgpath).get_data().max(axis=0).max(axis=0).max(axis=0)
mri = np.array(nib.load(imgpath).get_data()) / mx


def build_net():
    """Method for VGG like net Building.

    Returns
    -------
    nn : lasagne.layer
        Network.
    """
    nn = {}
    nn['input'] = InputLayer(inp_shape, input_var=input_var)

    nn['conv1a'] = Conv3DDNNLayer(nn['input'], 8, 3)
    nn['conv1b'] = Conv3DDNNLayer(nn['conv1a'], 8, 3, nonlinearity=identity)
    nn['nl1'] = NonlinearityLayer(nn['conv1b'])
    nn['pool1'] = Pool3DDNNLayer(nn['nl1'], 2)

    nn['conv2a'] = Conv3DDNNLayer(nn['pool1'], 16, 3)
    nn['conv2b'] = Conv3DDNNLayer(nn['conv2a'], 16, 3, nonlinearity=identity)
    nn['nl2'] = NonlinearityLayer(nn['conv2b'])
    nn['pool2'] = Pool3DDNNLayer(nn['nl2'], 2)

    nn['conv3a'] = Conv3DDNNLayer(nn['pool2'], 32, 3)
    nn['conv3b'] = Conv3DDNNLayer(nn['conv3a'], 32, 3)
    nn['conv3c'] = Conv3DDNNLayer(nn['conv3b'], 32, 3, nonlinearity=identity)
    nn['nl3'] = NonlinearityLayer(nn['conv3c'])
    nn['pool3'] = Pool3DDNNLayer(nn['nl3'], 2)

    nn['conv4a'] = Conv3DDNNLayer(nn['pool3'], 64, 3)
    nn['conv4b'] = Conv3DDNNLayer(nn['conv4a'], 64, 3)
    nn['conv4c'] = Conv3DDNNLayer(nn['conv4b'], 64, 3, nonlinearity=identity)
    nn['nl4'] = NonlinearityLayer(nn['conv4c'])
    nn['pool4'] = Pool3DDNNLayer(nn['nl4'], 2)

    nn['dense1'] = DenseLayer(nn['pool4'], num_units=128)
    nn['bn'] = BatchNormLayer(nn['dense1'])
    nn['dropout'] = DropoutLayer(nn['bn'], p=0.7)

    nn['dense2'] = DenseLayer(nn['dropout'], num_units=64)
    
    #nn['pool4'] = GlobalPoolLayer(nn['nl4'])

    nn['prob'] = DenseLayer(nn['dense2'], num_units=2,
                    nonlinearity=lasagne.nonlinearities.softmax)
    return nn


net = build_net()

# Load parameters
with np.load(modelname) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(net['prob'], param_values)



def get_train_functions(net):
	test_prediction = lasagne.layers.get_output(net['prob'], deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
	test_loss = test_loss.mean()

	val_fn = theano.function([input_var, target_var], test_loss)
	pred_fn = theano.function([input_var], test_prediction)

	return val_fn, pred_fn

val_fn, pred_fn = get_train_functions(net)
test_loss = val_fn(mri.reshape(1,1,110, 110, 110), np.array(np.int32(1)).reshape((1)))
pred = pred_fn(mri.reshape(1,1,110, 110, 110))

print(pred)
print("test_loss",test_loss)


#you can change to the layer that you want to visualize
outlayer, activation = lasagne.layers.get_output([net['prob'],net['conv4c']],deterministic=True)


test_loss = lasagne.objectives.categorical_crossentropy(outlayer, target_var)
test_loss = test_loss.mean()
grad = T.grad(test_loss, activation)


test_fn = theano.function([input_var], outlayer)
grad_fn = theano.function([input_var,target_var], grad)
activation_fn = theano.function([input_var], activation)


grad_val = grad_fn(np.array(mri).reshape((-1, 1, 110, 110, 110)),np.array(np.int32(1)).reshape((1)))

activation_val = activation_fn(np.array(mri).reshape((-1, 1, 110, 110, 110)))[0]

class_weights = np.sum(grad_val, axis=(2,3,4)) / np.prod(grad_val.shape[2:5])

cam = np.zeros(dtype = np.float32, shape = activation_val.shape[1:4])



for j, w in enumerate(class_weights[0,:]):
    cam += w * activation_val[j, :, :, :]


cam_name = 'npz_vgg/' + prefix + '_cam_conv4c.npz'
mri_name = 'npz_vgg/' + prefix + '_mri.npz'


np.savez(cam_name, np.array(cam))
np.savez(mri_name, np.array(mri))
