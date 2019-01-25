#!/usr/bin/python3
import cv2
import numpy as np
import sys, os
import json
import traceback
from pynput import mouse
import argparse
import matplotlib.pyplot as plt
import matplotlib
import threading
import queue
import time

minFrames = 30      # minimum amount of frames to capture per coordinate
aparser = argparse.ArgumentParser()
aparser.add_argument('-tSet', help='Choose training set')
aparser.add_argument('-outputDir', help='Choose directory to save training data')
aparser.add_argument('-view', help='Visualise training data', action='store_true')
aparser.add_argument('-fs', help='Full screen training: supply display config (e.g. xx/xx for 4 displays in 2 rows)')
aparser.add_argument('-save', help='Save config into cfg.json', action='store_true')
args = aparser.parse_args()

# read / write cfg file
if os.path.isfile('cfg.json'):
    with open('cfg.json', 'r') as cfgfile:
        cfg = json.load(cfgfile)
else:
    cfg = {}
if args.tSet or 'tSet' not in cfg:
    cfg['tSet'] = args.tSet
if args.outputDir or 'capturepath' not in cfg:
    cfg['capturepath'] = args.outputDir
if args.save:
    with open('cfg.json', 'w') as cfgfile:
        cfgfile.write(json.dumps(cfg, sort_keys=True, indent=2))

outputDir = os.path.realpath(cfg['capturepath'])
trainingFile = os.path.join(outputDir, '{}.json'.format(cfg['tSet']))
outputDataDir = os.path.join(outputDir, cfg['tSet'])

# read database file:
if os.path.isfile(trainingFile):
    with open(trainingFile) as jsonfile:
        trainingData = json.load(jsonfile)

    if args.view:
        X = np.asarray([k['x'] for k in trainingData])
        Y = np.asarray([k['y'] for k in trainingData])
        fig, axes = plt.subplots(2, 1, constrained_layout=True)
        axes[0].hist2d(X, Y, (50, 50), cmap=plt.cm.jet, norm = matplotlib.colors.LogNorm())
        axes[0].set_title('log hist')
        axes[1].scatter(X, Y, marker = 'x')
        axes[1].set_title('{} samples'.format(len(trainingData)))
        for ax in axes:
            ax.set_xlabel('pixel vertical')
            ax.set_ylabel('pixel horizontal')
        fig = plt.gcf()
        fig.canvas.set_window_title(cfg['tSet'])
        plt.show()
        sys.exit()
else:
    trainingData = []


def savefile():
    # save json file
    with open(trainingFile, 'w') as jsonfile:
        jsonfile.write(json.dumps(trainingData,
                                    sort_keys = True,
                                    indent = 2))
        print('saved as ' + trainingFile)
    
if args.fs:
    # show fs training app
    import tkinter as tk
    print('Use arrowkeys to navigate to a node.')
    print('Use <ENTER> to start recording at this node.')
    print('Use <SPACE> to switch to next screen.')
    print('Once a node is green enough frames have been recorded!')
    print('Recording can be resumed any time.')

    capQ = queue.Queue(20)
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
        def record(self, x, y):
            self.x = x
            self.y = y
            self.recording = True
            self.frames = 0

        def stopRecording(self):
            self.recording = False

        def run(self):
            self.capture = True
            self.recording = False
            if not os.path.exists(outputDataDir):
                os.makedirs(outputDataDir)
            while self.capture:
                start = time.time()
                frame = capQ.get()
                if self.recording:
                    self.frames += 1
                    fname = 'cap-{:06d}.jpg'.format(len(trainingData))
                    cv2.imwrite(os.path.join(outputDataDir, fname), frame)
                    trainingData.append({'x': self.x,
                                         'y': self.y,
                                         'f': fname})
                    print(' -> ' + os.path.join(outputDataDir, fname) + ' ({:5.2f} fps)'.format(1.0 / (time.time() - start)))

        def stop(self):
            self.capture = False

    class gPoint(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.frames = 0

    class xyz(object):
        def __init__(self):
            self.cThread = CaptureThread()
            self.cThread.start()
            self.cHandler = CaptureHandler()
            self.cHandler.start()

            self.root = tk.Tk()
            self.canv = tk.Canvas(self.root)
            self.canv.pack()
            self.width, self.height = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
            self.currPosX = 0
            self.currPosY = 0
            self.offsetX = 0
            self.offsetY = 0
            self.cwidth = 0
            self.cheight = 0
            self.incCol = False
            self.incRow = False
            self.killed = False
            self.recording = False
            self.root.attributes('-fullscreen', True)
            self.root.after(100, self.root.quit)
            self.root.mainloop()
            # now we have our display dimensions
            self.updGeom()
            self.drawCurr()
            self.bindKeys()

        def bindKeys(self):
            self.root.bind('<Escape>', self.kill)
            self.root.bind('<Left>', self.moveL)
            self.root.bind('<Right>', self.moveR)
            self.root.bind('<Up>', self.moveU)
            self.root.bind('<Down>', self.moveD)
            self.root.bind('<Return>', self.record)
            self.root.bind('<space>', self.nextw)
        
        def unbindKeys(self):
            self.root.unbind('<Escape>')
            self.root.unbind('<Left>')
            self.root.unbind('<Right>')
            self.root.unbind('<Up>')
            self.root.unbind('<Down>')
            self.root.unbind('<space>')

        def kill(self, ev):
            self.cThread.stop()
            self.cHandler.stop()
            self.killed = True
            self.root.destroy()
    
        def moveR(self, ev):
            if self.currPosX < len(self.grid) - 1:
                self.currPosX += 1
                self.drawCurr()

        def moveL(self, ev):
            if self.currPosX > 0:
                self.currPosX -= 1
                self.drawCurr()

        def moveU(self, ev):
            if self.currPosY > 0: 
                self.currPosY -= 1
                self.drawCurr()

        def moveD(self, ev):
            if self.currPosY < len(self.grid[0]) - 1:
                self.currPosY += 1
                self.drawCurr()

        def drawCurr(self):
            self.canv.delete('all')
            self.drawXs()
            currpt = self.grid[self.currPosX][self.currPosY]
            self.drawX(currpt.x, currpt.y, 'blue', 3.)
    
        def updGeom(self):
            self.offsetX += self.cwidth if self.incCol else 0
            self.offsetY += self.cheight if self.incRow else 0
            self.cwidth, self.cheight = [int(k) for k in self.root.geometry().split('+')[0].split('x')]
            self.canv.config(width=self.cwidth, height=self.cheight)
            rangeY = range(20, self.cheight + 10, 100)
            rangeX = range(20, self.cwidth + 10, 100)
            self.grid = [[gPoint(x, y) for y in rangeY] for x in rangeX]
            if os.path.exists(trainingFile):
                with open(trainingFile) as tf:
                    content = json.load(tf)
                for data in content:
                    if data['x'] - self.offsetX in rangeX and data['y'] - self.offsetY in rangeY:
                        self.grid[rangeX.index(data['x'] - self.offsetX)][rangeY.index(data['y'] - self.offsetY)].frames += 1
            self.drawXs()
        
        def drawX(self, x, y, col='black', width = 1.):
            self.canv.create_oval(x - 1, y - 1, x + 1, y + 1, fill=col)
            self.canv.create_line(x - 20, y - 20, x + 20, y + 20, fill=col, width=width)
            self.canv.create_line(x - 20, y + 20, x + 20, y - 20, fill=col, width=width)

        def drawXs(self):
            for rows in self.grid:
                for pt in rows:
                    if pt.frames >= minFrames:
                        self.drawX(pt.x, pt.y, col='green', width = 2.)
                    else:
                         self.drawX(pt.x, pt.y, col='grey')

        def record(self, ev):
            if not self.recording:
                self.recording = True
                self.unbindKeys()
                currpt = self.grid[self.currPosX][self.currPosY]
                self.drawX(currpt.x, currpt.y, 'red', 3.)
                self.cHandler.record(currpt.x + self.offsetX, currpt.y + self.offsetY)
            else:
                self.recording = False
                self.cHandler.stopRecording()
                self.grid[self.currPosX][self.currPosY].frames += self.cHandler.frames
                self.drawCurr()
                self.bindKeys()

        def nextw(self, ev):
            self.currPosX = self.currPosY = 0
            self.root.geometry('{}x{}+{}+{}'.format(self.width, self.height, 
                int((self.ridx - .5)*self.width/self.row), int((self.cidx - .5)*self.height/len(self.disprows))))
            self.updGeom()
            self.drawCurr()
            self.root.quit()  

        def run(self):
            self.disprows = [len(k) for k in args.fs.split('/')]
            for ridx, row in enumerate(self.disprows, 1):
                self.ridx = ridx
                self.row = row
                if self.killed:
                    break
                for cidx in range(1, row + 1):
                    self.cidx = cidx
                    self.incCol = True
                    # root.attributes('-zoomed', True)
                    self.root.attributes('-fullscreen', True)
                    self.root.mainloop()
                    self.incCol = False
                    self.incRow = False
                    if self.killed:
                        break
                self.incRow = True

    xyz().run()
    savefile()

else:
    # configure capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print('Error opening device!')
        sys.exit()

    def handleClick(x, y, button, pressed):
        # will not trigger on any wayland window
        if pressed:
            sys.stdout.write('mouseDown @ {}, button {}'.format((x, y), button))
            if button == mouse.Button.left:
                # capture cam
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    if not os.path.exists(outputDataDir):
                        os.makedirs(outputDataDir)

                    fname = 'cap-{:06d}.jpg'.format(len(trainingData))
                    cv2.imwrite(os.path.join(outputDataDir, fname), frame)
                    trainingData.append({'x': x,
                                         'y': y,
                                         'f': fname})
                    print(' -> ' + fname)
                else:
                    print(' -> discarded')

            else:
                print()

    with mouse.Listener(on_click = handleClick) as listener:
        try:
            print('Listening for left-clicks...')
            listener.join()
        except:
            print()
            traceback.print_exc()
            savefile()


    cap.release()