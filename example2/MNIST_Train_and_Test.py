
# coding: utf-8

# # The code comes from [MNIST 的手寫數字辨識](https://clay-atlas.com/blog/2019/09/28/%E4%BD%BF%E7%94%A8-cnn-%E9%80%B2%E8%A1%8C-mnist-%E7%9A%84%E6%89%8B%E5%AF%AB%E6%95%B8%E5%AD%97%E8%BE%A8%E8%AD%98-by-keras-%E5%AF%A6%E6%88%B0%E7%AF%87/)

# In[1]:


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[2]:


import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils, plot_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import backend as K


# In[3]:


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
x_train = X_train.reshape(60000, 1, 28, 28)/255
x_test = X_test.reshape(10000, 1, 28, 28)/255
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)


# In[4]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, input_shape=(1, 28, 28), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
print(model.summary())


# In[7]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=1)
loss, accuracy = model.evaluate(x_test, y_test)
print('Test:')
print('Loss: %s\nAccuracy: %s' % (loss, accuracy))


# In[8]:


model.save('./CNN_Mnist.h5')


# In[13]:


model = load_model('./CNN_Mnist.h5')


# In[19]:


n=80


# In[27]:


for n in range(0,10):
    # Display
    def plot_img(n):
        plt.imshow(X_test[n], cmap='gray')
        plt.show()


    predict = model.predict_classes(x_test)
    print('Prediction:', predict[n])
    print('Answer:', Y_test[n])
    plot_img(n)

