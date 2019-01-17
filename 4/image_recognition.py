# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import print_function

import keras
from keras.application.vgg16 import VGG16
from keras.models import  Sequential , Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import losses

from keras.preprocessing import image
import glob
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from keras.layers import Conv2D, MaxuPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json

import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def load_GE(): 
    imlist = []
    labels = []
    
    C1files = glob.glob(r'''D:\TUT\aspl\4\files\*.jpg''' )
    print('Data len: ', len((C1files)))
    
    # Load all images
    # make labels
    # Convert class vectors to binary class matrices

    labels = np.loadtxt(r'''D:\TUT\aspl\4\labels.txt''' )   
    labels = np_utils.to_categorical(labels,2)
#    with open(r'''D:\TUT\aspl\4\labels.txt''') as f:
#       for line in f:
#           labels.append(str(line[0]))
                
    # make images list
    for file in C1files:
        img = np.array(image.load_img(file))
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imlist.append((img-np.amin(img))/np.amax(img))
#        imlist.append((img/255.0))
        
    return imlist,labels

X,y = load_GE()
num_classes = 2

# Data Generator and Augmentations
datag = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
#    horizontal_flip = True,                                                                                                                                                                                                                                                                                                                                                                                                                 kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk                                                           
#    birghtness_range=(0.5, 1.2),
#    fill_mode='nearest'
)

datag.fit(X_train)

#shuffle dataset randomly
data = list(zip(imlist, labels))
random.shuffle(data)
imlist, labels = zip(*data)

imlist = np.array(imlist)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X,y)

# model construction
model = Sequential()

N = 32
w,h = 5, 5
model.add(Conv2D(N, (w, h),
           input_shape=(64, 64, 3),
           activation = 'relu',
           padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (w, h),
           activation = 'relu',
           padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (w, h),
           activation = 'relu',
           padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())    

# learning 
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics = ['accuracy'])
epoch_counter = 1
while True:
    print(' * Epoch ' + str(epoch_counter) + ' * ') 
    
    for X_batch, y_batch in datag.flow(X_train, y_train, batch_size=X_train.shape[0]):
        X_batch = X_batch/255.0
       
        model.fit(X_train, y_train, batch_size,
              epochs=1,
              verbose=1,
              validation_split=0.15,
              shuffle=True )
        break
    
    acc = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy [loss, acc]: ' + str(acc))
    if acc[1] > 0.90 or epoch_counter >= 50:
        break 
        
    epoch_counter += 1

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

# Evaluation
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(optimizer='sgd',loss='binary_crossentropy',metrics = ['accuracy'])
loaded_model.load_weights("model.h5")

score = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

confusion_matrix = np.zeros((2,2))

for X, y in zip(X_test, y_test):

    X = X.reshape(64,64,3)
    result = loaded_model.predict(X)
    confusion_matrix[np.argmax(y), np.argmax(result[0])] += 1
    
print('Confusion Matrix'+confusion_matrix)

# Results Visualization
for i, (X, y)in enumerate(zip(X_test, y_test)):
    plt.imshow(X)
    plt.show()
    print('prediction: ', model.predict(X.reshape(64,64,3)))
    print('ground true:', y)
    if i == 20:
        break
    
# Play Video

def logVideoMetadata(video):

    print('current pose: ' + str(video.get(cv2.CAP_PROP_POS_MSEC)))
    print('0-based index: ' + str(video.get(cv2.CAP_PROP_POS_FRAMES)))
    print('pose: ' + str(video.get(cv2.CAP_PROP_POS_AVI_RATIO)))
    print('width: ' + str(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('height: ' + str(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('fps: ' + str(video.get(cv2.CAP_PROP_FPS)))
    print('codec: ' + str(video.get(cv2.CAP_PROP_FOURCC)))
    print('frame count: ' + str(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('format: ' + str(video.get(cv2.CAP_PROP_FORMAT)))
    print('mode: ' + str(video.get(cv2.CAP_PROP_MODE)))
    print('brightness: ' + str(video.get(cv2.CAP_PROP_BRIGHTNESS)))
    print('contrast: ' + str(video.get(cv2.CAP_PROP_CONTRAST)))
    print('saturation: ' + str(video.get(cv2.CAP_PROP_SATURATION)))
    print('hue: ' + str(video.get(cv2.CAP_PROP_HUE)))
    print('gain: ' + str(video.get(cv2.CAP_PROP_GAIN)))
    print('exposure: ' + str(video.get(cv2.CAP_PROP_EXPOSURE)))
    print('convert_rgb: ' + str(video.get(cv2.CAP_PROP_CONVERT_RGB)))
    print('rect: ' + str(video.get(cv2.CAP_PROP_RECTIFICATION)))
    print('iso speed: ' + str(video.get(cv2.CAP_PROP_ISO_SPEED)))
    print('buffersize: ' + str(video.get(cv2.CAP_PROP_BUFFERSIZE)))

def hot_ent_to_text(prediction):
    print(prediction)
    if(prediction[0,0] > prediction[0,1]):
        return 'NON-SMILE'
    else:
        return 'SMILE'

video = cv2.VideoCapture()
video_path = './smile_movie.MOV'
video.open(video_path)
if not video.isOpened():
    print('Error: unable to open video: ' + video_path)

logVideoMetadata(video)

resize_ratio = 0.125
roi = [150,550,800,800]
blur_kernel_size = 5

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) )
for i in range(total_frames):
    
    ret, orig_img = video.read()
    
    if i%20 != 0:
        continue
    
    img = orig_img[roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]]
    img = cv2.blur(img, (blur_kernel_size,blur_kernel_size))
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip( img, 0 )
    
    plt.imshow(img)
    plt.show()
    
    prediction = model.predict(img.reshape(1,64,64,3))
    print(hot_ent_to_text(prediction))
    print(30*'*')
