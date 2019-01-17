# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import losses

from keras.preprocessing import image
import glob
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
import cv2

def load_GE():
    imlist = []
    
    C1files = glob.glob(r'''D:\TUT\aspl\project3\files\*.jpg''' )
    for file in C1files:
# util function to convert a tensor into a valid image
        img = np.array(image.load_img(file))
        img = load_img(image_path)
        if img_size:
            scale = float(img_size) / max(img.size)
            new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
            img = img.resize(new_size, resample=Image.BILINEAR)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg16.preprocess_input(img)
        imlist.append((img-np.amin(img))/np.amax(img))
        
    return imlist 

X = load_GE()
#X = normalize(X)
y = np.loadtxt(r'''D:\TUT\aspl\project3\labels.txt''' )   
y = np_utils.to_categorical(y,2)
X_train, X_test, y_train, y_test = train_test_split(X,y)


base_model = VGG16(include_top=False, weights = "imagenet",
                   input_shape = (64,64,3))


N = 64
w,h = 5,5
m = base_model.output
m = (Conv2D(N, (w, h),
    activation = 'relu',
    padding = 'same'))(m)
m = MaxPooling2D(pool_size=(2, 2))(m)
m = (Conv2D(32, (w, h),
    activation = 'relu',
    padding = 'same'))(m)

m = Flatten()(m)
m = Dense(128, activation = 'relu')(m)
output = Dense(1, activation ='sigmoid')(m)
model = Model(inputs = [base_model.input], outputs = [output])

model.layers[-3].trainable = False
model.layers[-2].trainable = False
model.layers[-1].trainable = False

print(model.summary())
    
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics = ['accuracy'])
model.fit(X_train,y_train,batch_size=32,epochs=50,validation_data=[X_test,y_test])

