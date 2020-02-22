#!/usr/bin/python3.5
# coding: utf-8

#This bash script is for 3D-ResNet classification
#author:Qi Li
#github:liqi814
##Without Singularity
##python3.5 scripts/res_3d_pred.py > resnet.out &
##With Singularity
##nohup singularity exec --nv classify.img python3.5  scripts/res_3d_pred.py >resnet.out &

import os
import gc
import sys
import time
import datetime
import traceback
#from collections import OrderedDict

import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.utils.multiclass import unique_labels
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.cross_selection import train_test_split
#from sklearn.model_selection  import StratifiedKFold

import lasagne
import theano
from lasagne.layers import InputLayer
from lasagne.layers.dnn import Conv3DDNNLayer
from lasagne.layers.dnn import Pool3DDNNLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import identity, softmax
from lasagne.layers import DropoutLayer
import theano.tensor as T
#import pickle
#from skimage.transform import rotate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#from matplotlib.pyplot import imshow
#import copy
#import glob
#import nibabel
#from skimage.transform import resize
#import seaborn as sns


PATH_TO_REP = 'data/'  # adni_data
input_var = T.tensor5(name='input', dtype='float32')
target_var = T.ivector()
inp_shape = (None, 1, 110, 110, 110)


# In[3]:


modelname = 'models/pretrained_resnet.npz'


from utils import iterate_minibatches, iterate_minibatches_train


# ### Train functions

# In[ ]:
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_train_functions(nn, updates_method=lasagne.updates.nesterov_momentum,_lr=0.00001):
    """
    Return functions for training, validation network and predicting answers.

    Parameters
    ----------
    nn : lasagne.Layer
        network last layer

    updates_method : function
        like in lasagne.objectives or function from there

    _lr : float
        learning rate which relate with the updates_method

    Returns
    -------
    train_fn : theano.function
        Train network function.
    val_fn : theano.function
        Validation function.
    pred_fn : theano.function
        Function for get predicts from network.
    """
    prediction = lasagne.layers.get_output(nn['prob'])
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(nn['prob'], trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=_lr)

    test_prediction = lasagne.layers.get_output(nn['prob'], deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)
    pred_fn = theano.function([input_var], test_prediction)

    return train_fn, val_fn, pred_fn


# In[4]:
def train(train_fn, val_fn, test_fn,
          X_train, y_train,
          X_test, y_test,
          LABEL_1, LABEL_2,  # labels of the y.
          num_epochs=80, batchsize=4,
          dict_of_paths={'output': '1.txt', 'picture': '1.png',
                         'report': 'report.txt'},
          report='''trained next architecture, used some
                    optimizstion method with learning rate...''',
          architecture='nn=...'):
    """
    Iterate minibatches on train subset and validate results on test subset.

    Parameters
    ----------
    train_fn : theano.function
        Train network function.
    val_fn : theano.function
        Validation network function.
    test_fn : theano.function
        Function for get predicts from network.
    X_train : numpy array
        X train subset.
    y_train : numpy array
        Y train subset.
    X_test : numpy array
        X test subset.
    y_test : numpy array
        Y test subset.
    LABEL_1 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 0.
    LABEL_2 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 1.
    dict_of_paths : dictionary
        Names of files to store results.
    report : string
        Some comments which will saved into report after ending of training.
    num_epochs : integer
        Number of epochs for all of the experiments. Default is 50.
    batchsize : integer
        Batchsize for network training. Default is 5.

    Returns
    -------
    tr_losses : numpy.array
        Array with loss values on train.
    val_losses : numpy.array
        Array with loss values on test.
    val_accs : numpy.array
        Array with accuracy values on test.
    rocs : numpy.array
        Array with roc auc values on test.

    """

    eps = []
    tr_losses = []
    val_losses = []
    val_accs = []
    rocs = []

    FILE_PATH = dict_of_paths['output']
    PICTURE_PATH = dict_of_paths['picture']
    REPORT_PATH = dict_of_paths['report']

    # here we written outputs on each step (val and train losses, accuracy, auc)
    with open(FILE_PATH, 'w') as f:
        f.write('\n----------\n\n' + str(datetime.datetime.now())[:19])
        f.write('\n' + LABEL_1 + '-' + LABEL_2 + '\n')
        f.close()

    # starting training
    print("Starting training...", flush=True)
    #den = X_train.shape[0] / batchsize
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches_train(X_train, y_train, batchsize,shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_batches = 0
        preds = []
        targ = []
        for batch in iterate_minibatches(X_test, y_test, batchsize,
                                         shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
            out = test_fn(inputs)
            [preds.append(i) for i in out]
            [targ.append(i) for i in targets]

        preds_tst = np.array(preds).argmax(axis=1)
        ##
        ## output
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1,
                                                   num_epochs,
                                                   time.time() - start_time),
              flush=True)
        print("  training loss:\t\t{:.7f}".format(train_err / train_batches),
              flush=True)
        print("  validation loss:\t\t{:.7f}".format(val_err / val_batches),
              flush=True)
        print('  validation accuracy:\t\t{:.7f}'.format(
            accuracy_score(np.array(targ),
                           preds_tst)), flush=True)
        print('Confusion matrix for test:', flush=True)
        print(confusion_matrix(np.array(targ), np.array(preds).argmax(axis=1)),
              flush=True)
        rcs = roc_auc_score(np.array(targ), np.array(preds)[:, 1])
        sys.stderr.write('Pairwise ROC_AUCs: ' + str(rcs))
        print('')
        
        
        np.set_printoptions(precision=2)
        class_names=np.array(['AD', 'Normal'], dtype='<U10')
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(np.array(targ), np.array(preds).argmax(axis=1), classes=class_names,
                      title='Confusion matrix, without normalization')

        plt.savefig(results_folder + 'confusion_matrix.png')
        # Plot normalized confusion matrix
        plot_confusion_matrix(np.array(targ), np.array(preds).argmax(axis=1), classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
        plt.savefig(results_folder + 'Normalized_confusion_matrix.png')


        with open(FILE_PATH, 'a') as f:
            f.write("\nEpoch {} of {} took {:.3f}s".format(epoch + 1,
                                                           num_epochs,
                                                           time.time() - start_time))
            f.write(
                "\n training loss:\t\t{:.7f}".format(train_err / train_batches))
            f.write(
                "\n validation loss:\t\t{:.7f}".format(val_err / val_batches))
            f.write('\n validation accuracy:\t\t{:.7f}'.format(
                accuracy_score(np.array(targ),
                               np.array(preds).argmax(axis=1))))

            f.write('\n Pairwise ROC_AUCs:' + str(rcs) + '\n')
            f.close()
        ## output
        ## saving results
        eps.append(epoch + 1)
        tr_losses.append(train_err / train_batches)
        val_losses.append(val_err / val_batches)
        val_accs.append(
            accuracy_score(np.array(targ), np.array(preds).argmax(axis=1)))
        rocs.append(rcs)

    print('ended!')

    ### and save plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title('Loss ' + LABEL_1 + ' vs ' + LABEL_2)
    plt.xlabel('Epoch')
    plt.ylim((0, 3))
    plt.ylabel('Loss')
    plt.plot(eps, tr_losses, label='train')
    plt.plot(eps, val_losses, label='validation')
    plt.legend(loc=0)
    #
    plt.subplot(2, 2, 2)
    plt.title('Accuracy ' + LABEL_1 + ' vs ' + LABEL_2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(eps, val_accs, label='validation accuracy')
    plt.legend(loc=0)
    #
    plt.subplot(2, 2, 3)
    plt.title('AUC ' + LABEL_1 + ' vs ' + LABEL_2)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.plot(eps, np.array(rocs), label='validation auc')
    plt.legend(loc=0)
    #
    plt.subplot(2, 2, 4)
    plt.title('architecture')
    plt.axis('off')
    plt.text(0, -0.1, architecture, fontsize=7, )
    plt.savefig(PICTURE_PATH)
    ###########

    # write that trainig was ended
    with open(FILE_PATH, 'a') as f:
        f.write('\nended at ' + str(datetime.datetime.now())[:19] + '\n \n')
        f.close()

    # write report
    with open(REPORT_PATH, 'a') as f:
        f.write(
            '\n classification ' + LABEL_1 + ' vs ' + LABEL_2 + '\n' + report)
        #         f.write(architecture)
        f.write('final results are:')
        f.write('\n tr_loss: ' + str(tr_losses[-1]) + '\n val_loss: ' + str(val_losses[-1]) + '\n val_acc; ' + str(val_accs[-1]) + '\n val_roc_auc: ' + str(rocs[-1]))
        f.write('\nresults has been saved in files:\n')
        f.write(FILE_PATH + '\n')
        f.write(PICTURE_PATH + '\n')
        f.write('\n ___________________ \n\n\n')
        f.close()

    return tr_losses, val_losses, val_accs, rocs


def build_net():
    """Method for VGG like net Building.

    Returns
    -------
    nn : lasagne.layer
        Network.
    """
    net = {}
    net['input'] = InputLayer((None, 1, 110, 110, 110), input_var=input_var)
    net['conv1a'] = Conv3DDNNLayer(net['input'], 32, 3, pad='same',
                                   nonlinearity=identity)
    net['bn1a'] = BatchNormLayer(net['conv1a'])
    net['relu1a'] = NonlinearityLayer(net['bn1a'])
    net['conv1b'] = Conv3DDNNLayer(net['relu1a'], 32, 3, pad='same',
                                   nonlinearity=identity)
    net['bn1b'] = BatchNormLayer(net['conv1b'])
    net['relu1b'] = NonlinearityLayer(net['bn1b'])
    net['conv1c'] = Conv3DDNNLayer(net['relu1b'], 64, 3, stride=(2, 2, 2),
                                   pad='same', nonlinearity=identity)
    # VoxRes block 2
    net['voxres2_bn1'] = BatchNormLayer(net['conv1c'])
    net['voxres2_relu1'] = NonlinearityLayer(net['voxres2_bn1'])
    net['voxres2_conv1'] = Conv3DDNNLayer(net['voxres2_relu1'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres2_bn2'] = BatchNormLayer(net['voxres2_conv1'])
    net['voxres2_relu2'] = NonlinearityLayer(net['voxres2_bn2'])
    net['voxres2_conv2'] = Conv3DDNNLayer(net['voxres2_relu2'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres2_out'] = ElemwiseSumLayer([net['conv1c'],
                                           net['voxres2_conv2']])
    # VoxRes block 3
    net['voxres3_bn1'] = BatchNormLayer(net['voxres2_out'])
    net['voxres3_relu1'] = NonlinearityLayer(net['voxres3_bn1'])
    net['voxres3_conv1'] = Conv3DDNNLayer(net['voxres3_relu1'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres3_bn2'] = BatchNormLayer(net['voxres3_conv1'])
    net['voxres3_relu2'] = NonlinearityLayer(net['voxres3_bn2'])
    net['voxres3_conv2'] = Conv3DDNNLayer(net['voxres3_relu2'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres3_out'] = ElemwiseSumLayer([net['voxres2_out'],
                                           net['voxres3_conv2']])

    net['bn4'] = BatchNormLayer(net['voxres3_out'])
    net['relu4'] = NonlinearityLayer(net['bn4'])
    net['conv4'] = Conv3DDNNLayer(net['relu4'], 64, 3, stride=(2, 2, 2),
                                  pad='same', nonlinearity=identity)
    # VoxRes block 5
    net['voxres5_bn1'] = BatchNormLayer(net['conv4'])
    net['voxres5_relu1'] = NonlinearityLayer(net['voxres5_bn1'])
    net['voxres5_conv1'] = Conv3DDNNLayer(net['voxres5_relu1'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres5_bn2'] = BatchNormLayer(net['voxres5_conv1'])
    net['voxres5_relu2'] = NonlinearityLayer(net['voxres5_bn2'])
    net['voxres5_conv2'] = Conv3DDNNLayer(net['voxres5_relu2'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres5_out'] = ElemwiseSumLayer([net['conv4'], net['voxres5_conv2']])
    # VoxRes block 6
    net['voxres6_bn1'] = BatchNormLayer(net['voxres5_out'])
    net['voxres6_relu1'] = NonlinearityLayer(net['voxres6_bn1'])
    net['voxres6_conv1'] = Conv3DDNNLayer(net['voxres6_relu1'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres6_bn2'] = BatchNormLayer(net['voxres6_conv1'])
    net['voxres6_relu2'] = NonlinearityLayer(net['voxres6_bn2'])
    net['voxres6_conv2'] = Conv3DDNNLayer(net['voxres6_relu2'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres6_out'] = ElemwiseSumLayer([net['voxres5_out'],
                                           net['voxres6_conv2']])

    net['bn7'] = BatchNormLayer(net['voxres6_out'])
    net['relu7'] = NonlinearityLayer(net['bn7'])
    net['conv7'] = Conv3DDNNLayer(net['relu7'], 128, 3, stride=(2, 2, 2),
                                  pad='same', nonlinearity=identity)

    # VoxRes block 8
    net['voxres8_bn1'] = BatchNormLayer(net['conv7'])
    net['voxres8_relu1'] = NonlinearityLayer(net['voxres8_bn1'])
    net['voxres8_conv1'] = Conv3DDNNLayer(net['voxres8_relu1'], 128, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres8_bn2'] = BatchNormLayer(net['voxres8_conv1'])
    net['voxres8_relu2'] = NonlinearityLayer(net['voxres8_bn2'])
    net['voxres8_conv2'] = Conv3DDNNLayer(net['voxres8_relu2'], 128, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres8_out'] = ElemwiseSumLayer([net['conv7'], net['voxres8_conv2']])
    # VoxRes block 9
    net['voxres9_bn1'] = BatchNormLayer(net['voxres8_out'])
    net['voxres9_relu1'] = NonlinearityLayer(net['voxres9_bn1'])
    net['voxres9_conv1'] = Conv3DDNNLayer(net['voxres9_relu1'], 128, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres9_bn2'] = BatchNormLayer(net['voxres9_conv1'])
    net['voxres9_relu2'] = NonlinearityLayer(net['voxres9_bn2'])
    net['voxres9_conv2'] = Conv3DDNNLayer(net['voxres9_relu2'], 128, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres9_out'] = ElemwiseSumLayer([net['voxres8_out'],
                                           net['voxres9_conv2']])

    net['pool10'] = Pool3DDNNLayer(net['voxres9_out'], 7)
    net['fc11'] = DenseLayer(net['pool10'], 128)
    net['prob'] = DenseLayer(net['fc11'], 2, nonlinearity=softmax)
    
    return net


# In[5]:
net = build_net()
#test_prediction = lasagne.layers.get_output(net['prob'], deterministic=True)
#test_fn = theano.function([input_var], test_prediction)


# In[6]:


with np.load(modelname) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(net['prob'], param_values)


# In[7]:

LABEL_1 = 'AD'
LABEL_2 = 'Normal'
results_folder = './results_resnet/'


# writing architecture in report
architecture = '''

    net['input'] = InputLayer((None, 1, 110, 110, 110), input_var=input_var)
    net['conv1a'] = Conv3DDNNLayer(net['input'], 32, 3, pad='same',
                                   nonlinearity=identity)
    net['bn1a'] = BatchNormLayer(net['conv1a'])
    net['relu1a'] = NonlinearityLayer(net['bn1a'])
    net['conv1b'] = Conv3DDNNLayer(net['relu1a'], 32, 3, pad='same',
                                   nonlinearity=identity)
    net['bn1b'] = BatchNormLayer(net['conv1b'])
    net['relu1b'] = NonlinearityLayer(net['bn1b'])
    net['voxres5_out'] = ElemwiseSumLayer([net['conv4'], net['voxres5_conv2']])
    # VoxRes block 6
    net['voxres9_out'] = ElemwiseSumLayer([net['voxres8_out'],
                                           net['voxres9_conv2']])

    net['pool10'] = Pool3DDNNLayer(net['voxres9_out'], 7)
    net['fc11'] = DenseLayer(net['pool10'], 128)
    net['prob'] = DenseLayer(net['fc11'], 2, nonlinearity=softmax)
'''

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# reading data
gc.collect()
metadata = pd.read_csv(PATH_TO_REP + 'metadata.csv')
testdata = pd.read_csv(PATH_TO_REP + 'test.csv')

##training data read
smc_mask = ((metadata.Label == LABEL_1) | (metadata.Label == LABEL_2)).values.astype('bool')
y = (metadata[smc_mask].Label == LABEL_1).astype(np.int32).values
#data = np.zeros((smc_mask.sum(), 1, 182, 218, 182), dtype='float32')
data = np.zeros((smc_mask.sum(), 1, 110, 110, 110), dtype='float32')
# into memory
for it, im in tqdm(enumerate(metadata[smc_mask].Path.values),total=smc_mask.sum(), desc='Reading training MRI to memory'):
    mx = nib.load(im).get_data().max(axis=0).max(axis=0).max(axis=0)
    data[it, 0, :, :, :] = np.array(nib.load(im).get_data()) / mx

#testing data read
smc_mask = ((testdata.Label == LABEL_1) | (testdata.Label == LABEL_2)).values.astype('bool')
y_test = (testdata[smc_mask].Label == LABEL_1).astype(np.int32).values
#X_test = np.zeros((smc_mask.sum(), 1, 182, 218, 182), dtype='float32')
X_test = np.zeros((smc_mask.sum(), 1, 110, 110, 110), dtype='float32')
# into memory
for it, im in tqdm(enumerate(testdata[smc_mask].Path.values),total=smc_mask.sum(), desc='Reading testing MRI to memory'):
    mx = nib.load(im).get_data().max(axis=0).max(axis=0).max(axis=0)
    X_test[it, 0, :, :, :] = np.array(nib.load(im).get_data()) / mx

X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)


train_fn, val_fn, test_fn = get_train_functions(net,
                                                updates_method=lasagne.updates.adam,
                                                _lr=0.000027)

dict_of_paths = {
                'output': results_folder + 'Exp_CV_' + '_' + LABEL_1 + '_vs_' + \
                          LABEL_2 + '.txt',
                'picture': results_folder + 'Exp_CV_' + '_' + LABEL_1 + '_vs_' + \
                           LABEL_2 + '.png',
                'report': results_folder + 'Exp_CV_' + '_' + LABEL_1 + '_vs_' + LABEL_2 + '_report.txt'
            }

report = LABEL_1 + '_vs_' + LABEL_2 + 'cv_fold ' + 'adam, lr=0.000027' 


try:
    tr_losses, val_losses, val_accs, rocs = train(train_fn, val_fn,
                                                  test_fn, X_train,
                                                  y_train, X_val,
                                                  y_val, LABEL_1,
                                                  LABEL_2,
                                                  num_epochs=80,
                                                  batchsize=4,
                                                  dict_of_paths=dict_of_paths,
                                                  report=report,
                                                  architecture=architecture)
#    cv_results.append((tr_losses, val_losses, val_accs, rocs))
except Exception as e:
    with open('errors_msg.txt', 'a') as f:
        f.write('Time: ' + str(datetime.datetime.now())[:19] +'\n' + str(e) + traceback.format_exc())

np.savez('models/resnet_weights.npz',*lasagne.layers.get_all_param_values(net['prob']))


preds = []
targ = []
for it, img in enumerate(X_test):
    out = test_fn(img.reshape((1, 1, 110, 110, 110)))
    print(out.reshape(-1,),y_test[it])
    [preds.append(i) for i in out]
    [targ.append(y_test[it])]
	
preds_tst = np.array(preds).argmax(axis=1)

print('Test accuracy:\t\t{:.7f}'.format(
    accuracy_score(np.array(targ),
                           preds_tst)), flush=True)
						   
np.set_printoptions(precision=2)
class_names=np.array(['AD', 'Normal'], dtype='<U10')
# Plot non-normalized confusion matrix
plot_confusion_matrix(np.array(targ), np.array(preds).argmax(axis=1), classes=class_names,
          title='Confusion matrix without normalization')

plt.savefig(results_folder + 'Confusion_matrix_test.png')
# Plot normalized confusion matrix
plot_confusion_matrix(np.array(targ), np.array(preds).argmax(axis=1), classes=class_names, normalize=True,
        title='Normalized confusion matrix')
plt.savefig(results_folder + 'Normalized_confusion_matrix_test.png')

# In[ ]
#epochs = range(1,50)
'''
print("cv_results is {}".format(np.array(cv_results)))
#exit
plt.figure()
plt.plot(np.array(cv_results)[0, ], 'g', label='Training loss')
plt.plot(np.array(cv_results)[1, ], 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses.png')
'''
