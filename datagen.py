#!/usr/bin/python3
import sys
import os
import shutil
import argparse
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
from multiprocessing import Pool
import cv2
import json

aparser = argparse.ArgumentParser()
aparser.add_argument('-tSet', help='Choose source training set (inside capture-output)')
aparser.add_argument('-name', help='Output name of augmentation')
aparser.add_argument('-valPart', help='The percentage of data to use for validation (]0 < valPart < 1[)', default=.12)
aparser.add_argument('-copy', help='Only copy, do NOT modify training set', action='store_true')
aparser.add_argument('-subSetNum', help='Choose an absolute number of samples to use', default = -1)
aparser.add_argument('-pathCap', help='Specify path to capture-output', nargs=1)
aparser.add_argument('-pathAug', help='Specify path to augmentation-output', nargs=1)
aparser.add_argument('-save', help='Save config into cfg.json', action='store_true')
args = aparser.parse_args()

if os.path.exists('cfg.json'):
    with open('cfg.json', 'r') as cfgfile:
        cfg = json.load(cfgfile)
else:
    cfg = {}
if args.pathCap or 'capturepath' not in cfg:
    cfg['capturepath'] = args.pathCap
if args.pathAug or 'augpath' not in cfg:
    cfg['augpath'] = args.pathAug
if args.tSet or 'tSet' not in cfg:
    cfg['tSet'] = args.tSet
if args.name or 'nameOfRun' not in cfg:
    cfg['nameOfRun'] = args.name
if args.save:
    with open('cfg.json', 'w') as cfgfile:
        cfgfile.write(json.dumps(cfg, sort_keys=True, indent=2))

dataSet = cfg['tSet']
valPart = float(args.valPart)
subSetNum = int(args.subSetNum)
srcDirBare = os.path.join(cfg['capturepath'], dataSet)
srcDirTrain = os.path.join(cfg['augpath'], cfg['nameOfRun'] + '-train', 'images')
srcDirValidate = os.path.join(cfg['augpath'], cfg['nameOfRun'] + '-validate', 'images')

if os.path.exists(srcDirTrain):
    shutil.rmtree(srcDirTrain)
if os.path.exists(srcDirValidate):
    shutil.rmtree(srcDirValidate)

os.makedirs(srcDirTrain)
os.makedirs(srcDirValidate)

def randomBrightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = random.uniform(.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

datagen = ImageDataGenerator(rotation_range = 0,
                             width_shift_range = .05,
                             height_shift_range = .05,
                             brightness_range = (.005, 3),
                             horizontal_flip = False,
                             fill_mode = 'nearest')

files = [f for f in os.listdir(srcDirBare) if f.endswith('.jpeg') or f.endswith('.jpg')]
random.shuffle(files)
if args.subSetNum != -1:
    files = files[:int(args.subSetNum)]

filesTrain = files[:int(len(files) * (1 - valPart))]
filesValidate = files[int(len(files) * (1 - valPart)):]

if not args.copy:
    # gen data with first part
    def genImg(imgFile):
        img = load_img(os.path.join(srcDirBare, imgFile))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, 
                                batch_size = 1,
                                save_to_dir = srcDirTrain,
                                save_prefix = imgFile.split('.')[0],
                                shuffle = True,
                                save_format = 'jpeg'):
            i += 1
            if i > 20:
                break

    sys.stdout.write('Processing {} files...      '.format(len(filesTrain)))
    try:
        pool = Pool(os.cpu_count())
        for idx, _ in enumerate(pool.imap_unordered(genImg, filesTrain), 1):
            sys.stdout.write('\b' * 6 + '{:5.2f}%'.format(100 * float(idx) / len(filesTrain)))
            sys.stdout.flush()    
        print(' done!')

    finally:
        pool.close()
        pool.join()
else:
    sys.stdout.write('Copying {} files...'.format(len(filesTrain)))
    for imgFile in filesTrain:
        shutil.copy2(os.path.join(srcDirBare, imgFile), srcDirTrain)
    print(' done!')

# copy rest
sys.stdout.write('Copying {} files...'.format(len(filesValidate)))
for imgFile in filesValidate:
    shutil.copy2(os.path.join(srcDirBare, imgFile), srcDirValidate)
print(' done!')
