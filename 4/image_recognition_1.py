# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:24:05 2019

@author: he
"""
from __future__ import print_function



import sys
import glob
import os
print(sys.version)
print(sys.path)

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import keras
import random
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import losses
from keras import backend as K

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import glob
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from keras.models import model_from_json
import cv2


def load_GE():
    imlist = []
    labels = []
    
    C1files = glob.glob(r'''D:\TUT\TUT\aspl\4\files\*.jpg''' )
    for file in C1files:
# util function to convert a tensor into a valid image
#       img = np.array(image.load_img(file))
#        if img.size:
#            scale = (img-np.amin(img))/np.amax(img).astype(int)
#            print(scale)
#            new_size = (np.ceil(scale * int(img.size[0])).astype(int), np.ceil(scale * int(img.size[1])).astype(int)).astype(int)
#            img = img.resize(new_size, resample=image.BILINEAR)
#        img = np.expand_dims(img, axis=0)
#        img = vgg16.preprocess_input(img)
#        imlist.append(img)
        img = cv2.imread(file)
        img = (img-np.amin(img))/np.amax(img).astype('float32')
        img = cv2.resize(img, (64,64))
        img = img[...,::-1] #sort
        imlist.append(img) 
        
    with open(r'''D:\TUT\TUT\aspl\4\labels.txt''') as f:
        for line in f:
            labels.append(str(line[0]))
        
#    return imlist
    return imlist, labels   
        
X,y = load_GE()
#X = normalize(X)
#y = np.loadtxt(r'''D:\TUT\aspl\project3\labels.txt''' )   
y = np_utils.to_categorical(y,2)


c = list(zip(X,y))
random.shuffle(c)
X,y = zip(*c)
X = (np.array(X))
y = (np.array(y))

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size =0.8)

X_train2, X_validate, y_train2, y_validate = train_test_split(X_train,y_train)

# Data Generator and Augmentations

datag = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    #vertical_flip= True,
    brightness_range=(0.6, 1.0),
    fill_mode='nearest'
)

datag.fit(X_train2)

#base_model = VGG16(include_top=False, weights = "imagenet",
#                   input_shape = (64,64,3))

X_train2 = np.array(X_train2)
X_validate = np.array(X_validate)
y_train2 = np.array(y_train2)
y_validate = np.array(y_validate)

N = 64
w,h = 5,5
#m = base_model.output
model = Sequential()
model.add(Conv2D(N, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add((Conv2D(32, kernel_size=(w, h),activation = 'relu',padding = 'same')))
model.add(Dropout(0.2))

model.add((Conv2D(32, kernel_size=(w, h),activation = 'relu',padding = 'same')))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add((Conv2D(16, kernel_size=(w, h),activation = 'relu',padding = 'same')))
model.add(Dropout(0.2))

model.add((Conv2D(16, kernel_size=(w, h),activation = 'relu',padding = 'same')))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add((Conv2D(8, kernel_size=(w, h),activation = 'relu',padding = 'same')))
model.add(Dropout(0.2))

model.add (Flatten())
model.add (Dense(128, activation = 'relu'))
model.add (Dense(2, activation ='softmax'))

#model.layers[-3].trainable = False
#model.layers[-2].trainable = False
#model.layers[-1].trainable = False

print(model.summary())
    
# Learning Phase

model.compile(optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),loss='binary_crossentropy',metrics = ['accuracy'])

epoch_counter = 1
while True:
    
    print(' * Epoch ' + str(epoch_counter) + ' * ') 
    
    for x_batch, y_batch in datag.flow(X_train2, y_train2, batch_size=X_train2.shape[0]):
        x_batch = (x_batch-np.amin(x_batch))/np.amax(x_batch).astype('float32')
#        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        model.fit(X_train2,y_train2,batch_size=16,validation_data=[X_validate,y_validate])
        break
    val_acc = model.evaluate(X_validate, y_validate)
    print('Validation acc [loss, acc]: ' + str(val_acc))
    if val_acc[1] > 0.90 or epoch_counter >= 50:
        break 
         
    epoch_counter += 1

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

# Evaluation
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),loss='binary_crossentropy',metrics = ['accuracy'])
loaded_model.load_weights("model.h5")

model.fit(X_train,y_train,batch_size=16,validation_data=[X_test,y_test])

score = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

confusion_matrix = np.zeros((2,2))
cor = 0
performance = []
for x, y in zip(X_test, y_test):

    x = x.reshape(1,64,64,3)
    result = loaded_model.predict(x)
    confusion_matrix[np.argmax(y), np.argmax(result[0])] += 1
    if ((result[0,0] < result[0,1]) & (y[0] == 1)) | ((result[0,0]>result[0,1]) & (y[1]==0)):
        cor = cor+1
    performance.append(cor/800)
        
print('Confusion Matrix')
print(confusion_matrix)

# Results Visualization

for i, (x, y)in enumerate(zip(X_test, y_test)):
    plt.imshow(x)
    plt.show()
    
    print('prediction: ', loaded_model.predict(x.reshape(1,64,64,3)))

    print('ground true:', y)
    if i == 50:
        break
    
plt.plot(performance);
plt.show();
#  Play Video

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
    if(prediction[0,0] < prediction[0,1]):
        return 'NON-SMILE'
    else:
        return 'SMILE'


video = cv2.VideoCapture()
video.open(r'''D:\TUT\TUT\aspl\4\video.mp4''')
if not video.isOpened():
    print('Error: unable to open video: D:\TUT\TUT\aspl\4\video.mp4')

logVideoMetadata(video)

resize_ratio = 0.125
roi = [25,25,120,400]

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) )
for i in range(total_frames):
    
    ret, orig_img = video.read()
    
    if i%50 != 0:
        continue
    
    img = orig_img[roi[0]:roi[2],roi[1]:roi[3]]
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
#    img = np.asscalar(np.array(img).ravel())
#    img = (img-np.amin(img))/np.amax(img).astype('float32')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    plt.imshow(img)
    plt.show() 
    
    prediction = model.predict(img.reshape(1,64,64,3))
    print(hot_ent_to_text(prediction))
    print(30*'*')
