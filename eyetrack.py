#!/usr/bin/python3
import cv2
import numpy as np
import sys, os
import json
import argparse
import threading
import queue
import time

import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array


aparser = argparse.ArgumentParser()
aparser.add_argument('-tSet', help='Choose source training set (inside capture-output)')
aparser.add_argument('-name', help='Name of this particular run')
aparser.add_argument('-epoch', help='Model of which epoch to load (leave empty to select last)', nargs='?', const='')
aparser.add_argument('-path', help='Specify path to models', nargs=1)
aparser.add_argument('-view', help='Show processed frame', action='store_true')
aparser.add_argument('-save', help='Save config into cfg.json', action='store_true')
args = aparser.parse_args()

if os.path.exists('cfg.json'):
    with open('cfg.json', 'r') as cfgfile:
        cfg = json.load(cfgfile)
else:
    cfg = {}
if args.path or 'modelpath' not in cfg:
    cfg['modelpath'] = args.path
if args.tSet or 'tSet' not in cfg:
    cfg['tSet'] = args.tSet
if args.name or 'nameOfRun' not in cfg:
    cfg['nameOfRun'] = args.name
if args.save:
    with open('cfg.json', 'w') as cfgfile:
        cfgfile.write(json.dumps(cfg, sort_keys=True, indent=2))
    
modelPath = os.path.join(cfg['modelpath'], cfg['tset'], cfg['nameOfRun'], 'chkp')

if not args.epoch:
    lastModel = sorted(os.listdir(modelPath))[-1]
    modelPath = os.path.join(modelPath, lastModel)
else:
    modelPath = os.path.join(modelPath, '{:03d}.h5'.format(int(args.epoch) - 1))

capQ = queue.Queue(1)
class CaptureThread(threading.Thread):
    def run(self):
        # configure capture
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        self.capture = True
        if not cap.isOpened():
            print('Error opening device!')
            sys.exit()

        while self.capture:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                capQ.put(frame)
        cap.release()

    def stop(self):
        self.capture = False

class CaptureHandler(threading.Thread):
    def predict(self):
        self.predicting = True

    def stopPrediction(self):
        self.predicting = False

    def run(self):
        self.capture = True
        self.predicting = False
        self.model = load_model(modelPath)
        while self.capture:
            frame = capQ.get()
            if args.view:                
                cv2.imshow('current', frame)
                cv2.waitKey(25)
            if self.predicting:
                X = np.empty((1, 720, 1280, 3))
                x = img_to_array(frame)
                x.reshape((720, 1280, 3))
                x = x / 256.0
                X[0,] = x
                output = self.model.predict(X, None, verbose=1)[0]
                print('-> ({:4d}|{:4d})'.format(int(round(output[0] * 3840)), int(round(output[1] * 1080))))

    def stop(self):
        self.capture = False


try:
    cThread = CaptureThread()
    cThread.start()
    cHandler = CaptureHandler()
    cHandler.start()

    time.sleep(5)

    cHandler.predict()
except:
    cThread.stop()
    cHandler.stop()
finally:
    cHandler.join()
    cThread.join()
    cThread.stop()
    cHandler.stop()   