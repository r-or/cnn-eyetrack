#!/usr/bin/python3
import json
import pprint
import sys
import os
import numpy as np
import traceback
import random
import argparse
import json

import tensorflow
import keras
from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, History

from PIL import Image

# start with PYTHONHASHSEED=89
np.random.seed(44)
random.seed(22)
tensorflow.set_random_seed(11)

# session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1,
#                                       inter_op_parallelism_threads=1)
# tf_sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
# keras.backend.set_session(tf_sess)

pp = pprint.PrettyPrinter()

modes = ['train', 'predict', 'validate']

aparser = argparse.ArgumentParser()
aparser.add_argument('tSet', help='Choose source training set (inside capture-output)')
aparser.add_argument('mode', help=str(modes))
aparser.add_argument('name', help='Name of this particular run')
aparser.add_argument('-ls', help='List all current models', action='store_true')
aparser.add_argument('-useTdata', help='Use training data for prediction/validation instead of validation data', action='store_true')
aparser.add_argument('-pFile', help='prediction mode: name of the image file to predict')
args = aparser.parse_args()

trainingSet = args.tSet
mode = args.mode
nameofrun = args.name
predfile = args.pFile
srcT = args.useTdata

assert mode in modes

# paths
modelpath = '/media/cae/data2/models/'
if args.ls:
    print('available runs: ' + str(os.listdir(os.path.join(modelpath, trainingSet))))
    sys.exit()

outpath = os.path.join(modelpath, trainingSet, nameofrun)
modelPathBare = os.path.join(outpath, nameofrun)
cpmodelPathBare = os.path.join(outpath, 'chkp')
modelPath = modelPathBare + '.h5'

if not os.path.isdir(outpath):
    os.makedirs(outpath)
if not os.path.isdir(cpmodelPathBare):
    os.makedirs(cpmodelPathBare)

if len(os.listdir(cpmodelPathBare)):
    cpmodelPath = os.path.join(cpmodelPathBare, sorted(os.listdir(cpmodelPathBare))[-1])
    assert cpmodelPath.endswith('.h5')
else:
    cpmodelPath = None


if not os.path.isfile(modelPath) and not cpmodelPath:
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(720, 1280, 3)))
    model.add(LeakyReLU(alpha=.3))
    model.add(MaxPooling2D(pool_size=(2, 3)))

    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, kernel_initializer='random_uniform'))
    model.add(LeakyReLU(alpha=.3))
    #model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    adaD = optimizers.Adadelta()
    model.compile(loss='mse', optimizer=adaD)

    startEpoch = 0

else:
    # load model
    if os.path.isfile(modelPath):
        model = load_model(modelPath)
        # load training cfg
        if os.path.isfile(modelPathBare + '.json'):
            with open(modelPathBare + '.json', 'r') as jsonfile:
                modelcfg = json.load(jsonfile)
            startEpoch = modelcfg['epochsTrained']
        else:
            startEpoch = 0
    else:
        model = load_model(cpmodelPath)
        startEpoch = int(os.path.basename(cpmodelPath).split('.')[0])

scaleX = 1920 * 2
scaleY = 1080

with open(os.path.join('/media/cae/data2/capture-output', trainingSet + '.json')) as jsonfile:
    trainingdata = json.load(jsonfile)

dset = {}
for d in trainingdata:
    dset[d['f'].split('.')[0]] = (float(d['x']) / scaleX, 
                                float(d['y']) / scaleY)

tset = {}
tfiles = []
vset = {}
vfiles = []
trainDir = os.path.join('/media/cae/data2/augInput', trainingSet + '-train', 'images')
valDir = os.path.join('/media/cae/data2/augInput', trainingSet + '-validate', 'images')
for f in os.listdir(trainDir):
    tset[f] = dset[f.split('.')[0].split('_')[0]]
    tfiles.append(f)

for f in os.listdir(valDir):
    vset[f] = dset[f.split('.')[0]]
    vfiles.append(f)

batch_size = min(16, len(tfiles) // 16)
print('{} training samples, {} validation samples'.format(len(tfiles), len(vfiles)))
print(' -> Batch size chosen: {}'.format(batch_size))

class DataGen(keras.utils.Sequence):
    def __init__(self, filenames, path, labels, batchSize, dim, nChannels, shuffle=True):
        self.dim = dim
        self.batchSize = batchSize
        self.labels = labels
        self.filenames = filenames
        self.path = path
        self.nChannels = nChannels
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.filenames) / self.batchSize))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batchSize : (index + 1) * self.batchSize]
        fNamesTmp = [self.filenames[k] for k in indexes]
        X, y = self.__data_generation(fNamesTmp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, fNamesTmp):
        X = np.empty((self.batchSize, *self.dim, self.nChannels))
        Y = np.empty((self.batchSize, 2))
        for idx, fname in enumerate(fNamesTmp):
            img = load_img(os.path.join(self.path, fname))
            x = img_to_array(img)
            x.reshape((720, 1280, 3))
            x *=  1.0/256.0
            X[idx,] = x
            Y[idx,] = np.asarray(self.labels[fname])

        return X, Y

if mode == 'train':
    training_generator = DataGen(tfiles, trainDir, tset, batch_size, (720, 1280), 3, shuffle=True)
    validation_generator = DataGen(vfiles, valDir, vset, batch_size, (720, 1280), 3, shuffle=False)

    checkpointer = ModelCheckpoint(filepath=os.path.join(cpmodelPathBare, '{epoch:03d}.h5'), verbose=1, save_best_only=True)
    hist = History()
    try:
        model.fit_generator(training_generator,
                            steps_per_epoch=len(tfiles) // batch_size,
                            epochs=50,
                            validation_data=validation_generator,
                            validation_steps=len(vfiles) // batch_size,
                            max_queue_size=4, 
                            workers=4,
                            initial_epoch=startEpoch,
                            callbacks=[checkpointer, hist])
    except:
        print()
        traceback.print_exc()
    finally:
        print('hist: loss - validation loss')
        if 'loss' in hist.history:
            epochsTrained = len(hist.history['loss'])
            for l, vl in zip(hist.history['loss'], hist.history['val_loss']):
                print('{:.5f} - {:.5f}'.format(l, vl))
        else:
            print('N/A')
            epochsTrained = 0

        # always save your weights after training or during training
        model.save(modelPath)
        print('Saved model as "{}"'.format(modelPath))
        with open(modelPathBare + '.json', 'w') as jsonfile:
            jsonfile.write(json.dumps({'epochsTrained': epochsTrained + startEpoch}, sort_keys = True, indent = 2))

elif mode == 'predict':
    # print(model.summary())
    # pp.pprint(model.get_weights())
    X = np.empty((1, 720, 1280, 3))
    img = load_img(os.path.join(trainDir if srcT else valDir, sys.argv[4]))
    x = img_to_array(img)
    x.reshape((720, 1280, 3))
    x = x / 256.0
    X[0,] = x
    output = model.predict(X, None, verbose=1)[0]
    print('output: ({:.5f}, {:.5f}) - unscaled: ({:5.2f}, {:5.2f})'.format(output[0], output[1], output[0] * scaleX, output[1] * scaleY))
    exp = np.asarray(tset[predfile] if srcT else vset[predfile])
    print('expected: ({:.5f}, {:.5f}) - unscaled: ({:5.2f}, {:5.2f})'.format(exp[0], exp[1], exp[0] * scaleX, exp[1] * scaleY))

elif mode == 'validate':
    if srcT:
        files = tfiles
        validation_generator = DataGen(files, trainDir, tset, batch_size, (720, 1280), 3, shuffle=False)
    else:
        files = vfiles
        validation_generator = DataGen(files, valDir, vset, batch_size, (720, 1280), 3, shuffle=False)

    predictions = model.predict_generator(validation_generator, verbose=1)
    MSE = 0
    for f, pred in zip(files, predictions):
        exp = np.asarray(tset[f] if srcT else vset[f])
        mse = ((exp[0] - pred[0])**2 + (exp[1] - pred[1])**2) / 2
        print('{}: ({:.3f}, {:.3f}) -> ({:.3f}, {:.3f}) [mse: {:.3f}]'.format(f, exp[0], exp[1], pred[0], pred[1], mse))
        MSE += mse
    print('/MSE: {:.3f}'.format(MSE / len(files)))
