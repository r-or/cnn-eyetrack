# cnn-eyetrack
WIP eyetracker based on a standard CNN

### capture.py
Captures a webcam either
* each time a user left-clicks
* or a whole set of captures distributed on all the available screens (via tkinter-GUI)

### datagen.py
Creates a train/verification set from a set of captured & labelled images

### model.py
Trains/evaluates a model

### eyetrack.py
The finished eyetracker. Predicts the mouse location based on the input provided by the webcam.
