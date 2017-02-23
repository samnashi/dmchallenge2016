'''
Train DM Challenge classifier
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py <in:dataset> <out:trained_model>

'''
from __future__ import print_function
import numpy as np
import sys
import tables
import keras.backend as K
import json
import os
from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.core import Flatten, Dense, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from datetime import datetime

K.set_image_dim_ordering('th')
if K.image_dim_ordering() == 'th':
    channel_axis = 1

# training parameters
BATCH_SIZE = 100
NB_SMALL = 3500
NB_EPOCH_SMALL_DATA = 5
NB_EPOCH_LARGE_DATA = 50

# dataset
DATASET_BATCH_SIZE = 400

# global consts
EXPECTED_SIZE = 299
EXPECTED_CHANNELS = 3
EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
MODEL_PATH = 'model_{}.zip'.format(datetime.now().strftime('%Y%m%d%H%M%S'))

def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              name=None):
    '''Utility function to apply conv + BN.
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation='relu',
                      border_mode=border_mode,
                      name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x

def dataset_generator(dataset, batch_size):
    while True:
        i = 0
        while i < dataset.data.nrows:
            end = i + batch_size
            X = dataset.data[i:end]
            Y = dataset.labels[i:end]
            i = end
            yield(X, Y)


def confusion(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + K.epsilon())
    tn = K.sum(y_neg * y_pred_neg) / (K.sum(y_neg) + K.epsilon())
    return {'true_pos': tp, 'true_neg': tn}

# command line arguments
dataset_file = sys.argv[1]
model_file = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
verbosity = int(sys.argv[3]) if len(sys.argv) > 3 else 1

# loading dataset
print('Loading train dataset: {}'.format(dataset_file))
datafile = tables.open_file(dataset_file, mode='r')
dataset = datafile.root
#print(dataset.data[:].shape)

# determine training params based on data size
if dataset.data[:].shape[0] <= NB_SMALL:
    NB_EPOCH = NB_EPOCH_SMALL_DATA
else:
    NB_EPOCH = NB_EPOCH_LARGE_DATA

# set class_weight dynamically
ratio = dataset.ratio[0]
class_weight = {0: ratio[0], 1: ratio[1]}

print('BATCH_SIZE: {}'.format(BATCH_SIZE))
print('NB_EPOCH: {}'.format(NB_EPOCH))
print('class_weight: {}'.format(class_weight))

# setup model
print('Preparing model')
base_model = InceptionV3(weights='imagenet', include_top=False,input_tensor=None, input_shape=EXPECTED_DIM)

x = base_model.output  # base model is the inceptionV3
branch1x1 = conv2d_bn(x, 320, 1, 1)
branch3x3 = conv2d_bn(x, 384, 1, 1)
branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
branch3x3 = merge([branch3x3_1, branch3x3_2],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed11_interm')

branch3x3dbl = conv2d_bn(x, 448, 1, 1)
branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                     mode='concat', concat_axis=channel_axis)

branch_pool = AveragePooling2D(
    (3, 3), strides=(1, 1), border_mode='same')(x)
branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
          mode='concat', concat_axis=channel_axis,
          name='mixed11')

x = Convolution2D(1024, 1, 1, init='he_normal', activation='relu', border_mode='valid', subsample=(1, 1),
                  dim_ordering='th',
                  name='1x1Conv1_BC')(x)
x = BatchNormalization(mode=2, axis=1, name='BC_BN1')(x)
x = Convolution2D(1024, 2, 2, init='he_normal', activation='relu', border_mode='valid', subsample=(2, 2),
                  dim_ordering='th',
                  name='Conv2_BC')(x)
x = BatchNormalization(mode=2, axis=1, name='BC_BN2')(x)
x = Convolution2D(1024, 2, 2, init='he_normal', activation='relu', border_mode='valid', subsample=(2, 2),
                  dim_ordering='th', name='Conv3_BC')(x)
x = BatchNormalization(mode=2, axis=1, name='BC_BN3')(x)
x = Convolution2D(512, 1, 1, init='he_normal', activation='relu', border_mode='valid', subsample=(1, 1),
                  dim_ordering='th', name='1x1Conv2_BC')(x)
x = BatchNormalization(mode=2, axis=1, name='BC_BN4')(x)
x = Convolution2D(512, 2, 2, init='he_normal', activation='relu', border_mode='valid', subsample=(2, 2),
                  dim_ordering='th', name='Conv5_BC')(x)
x = BatchNormalization(mode=2, axis=1, name='BC_BN5')(x)
x = Convolution2D(256, 1, 1, init='he_normal', activation='relu', border_mode='valid', subsample=(1, 1),
                  dim_ordering='th', name='1x1Conv3_BC')(x)
x = BatchNormalization(mode=2, axis=1, name='BC_BN6')(x)
x = Flatten(name='BC_Flatten')(x)
x = BatchNormalization(mode=2, axis=1, name='BC_BN7')(x)
x = Dense(2048, init='he_normal',activation='relu', name='Dense_Final')(x)
x = BatchNormalization(mode=2, axis=1, name='BC_BN8')(x)
predictions = Dense(1, init='zero', activation='sigmoid', name='BC_Softmax')(x)

#this is the model we will train
model = Model(input=base_model.input, output=predictions)

# freeze base_model layers
for layer in model.layers:
    layer.trainable = False

#for layer in model.layers[216:249]: Atrous
for layer in model.layers[217:257]: #inception-fc conv
    layer.trainable = True

for i, layer in enumerate(model.layers):
	print(i, layer.name, "Trainable?: ", layer.trainable)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', confusion])

model.summary()

# early stopping
early_stopping_acc = EarlyStopping(monitor='loss', patience=30)

# training model
num_rows = dataset.data.nrows
print ("# of rows in dataset: ", num_rows)
half_num_rows = int(num_rows/2)
if num_rows > DATASET_BATCH_SIZE:
    # batch training
    model.fit_generator(
        dataset_generator(dataset, BATCH_SIZE),
        samples_per_epoch=num_rows,
        nb_epoch=NB_EPOCH,
        class_weight=class_weight,
        callbacks=[early_stopping_acc],
        verbose=2
    )

else:
    # one-go training
    X = dataset.data[:]
    Y = dataset.labels[:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)
    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,
              nb_epoch=NB_EPOCH,
              validation_data=(X_test, Y_test),
              shuffle=True,
              verbose=1,
              callbacks=[early_stopping_acc],
              class_weight=class_weight)

# saving model weights and architecture only
# to save space
print('Saving model')
model_name = os.path.basename(model_file)
model_path = os.path.splitext(model_file)[0]
weights_file = model_path + '.weights.h5'
arch_file = model_path + '.arch.json'
model.save_weights(weights_file)
with open(arch_file, 'w') as outfile:
    outfile.write(model.to_json())

# batch evaluate
# print('Evaluating')
# score = model.evaluate_generator(dataset_generator(dataset, BATCH_SIZE), num_rows)
# for i in range(1, len(model.metrics_names)):
#     print('{}: {}%'.format(model.metrics_names[i], score[i] * 100))

# close dataset
datafile.close()

print('Done.')
