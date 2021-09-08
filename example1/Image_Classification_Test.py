#!/usr/bin/env python
# coding: utf-8

# # This tutorial comes from [GeeksforGeeks | Image Classification using Keras](https://www.geeksforgeeks.org/python-image-classification-using-keras/)

# In[1]:


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)
from keras.models import load_model
from keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
import numpy as np


# In[2]:


img_width, img_height = 227, 227
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[3]:


model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[13]:


model2 = Sequential()
model2 = model.load_weights('model_saved.h5')
 
image2 = load_img('v_data/test/cars/5.jpg', target_size=(227, 227))
img2 = np.array(image2)
img2 = img2 / 255.0
img2 = img2.reshape(1,227,227,3)
label = model.predict(img2)
# label = model.predict_classes(img2)

# predict_x = model.predict(img2)
# label = np.argmax(predict_x,axis=1)
print("Predicted Class (0 - Cars , 1- Planes): ", label[0][0])


# In[ ]:




