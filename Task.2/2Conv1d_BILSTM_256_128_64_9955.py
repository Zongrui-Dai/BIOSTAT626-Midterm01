# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 05:45:50 2023

@author: dzr
"""
#################################################
##      Grid Search for Conv1d+BILSTM          ##
#################################################

import numpy as np
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM,Conv1D,MaxPooling1D
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
import pandas as pd
from keras.layers import Bidirectional
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback, LearningRateScheduler,ModelCheckpoint
from plot_keras_history import show_history, plot_history
from tensorflow.keras.utils import to_categorical
from plot_keras_history import show_history, plot_history
import matplotlib.pyplot as plt

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

training=pd.read_csv('E:/Biostatistics Master/BIOSTAT626/Midterm1/multiclass_training.csv')
testing=pd.read_csv('E:/Biostatistics Master/BIOSTAT626/Midterm1/testing.csv')

training = np.array(training)
testing = np.array(testing)

n_steps = 15
n_features = 561

def scheduler(epoch, lr=1e-2):
  if epoch < 500:
    return lr
  elif epoch < 700:
    return lr * tf.math.exp(-0.005)
  elif epoch < 1000:
    return lr * tf.math.exp(-0.01)
  elif epoch < 1500:
    return 5e-5
  else:
    return lr * tf.math.exp(-0.03)
checkpoint = ModelCheckpoint(filepath='E:/Biostatistics Master/BIOSTAT626/Midterm1//LSTM_grid/Conv1d_BILSTM_Complex.h5',
                                             monitor='val_acc',mode='auto' ,save_best_only='True')
    
callbacks_list = [                 
                    keras.callbacks.LearningRateScheduler(scheduler),
                    checkpoint
]


opt = tf.keras.optimizers.Adam(
    learning_rate=1e-3)
X, y = split_sequences(training, n_steps)
tX, ty = split_sequences(testing, n_steps)
    
trainy = to_categorical(y)

model = Sequential()
model.add(Conv1D(input_shape=(n_steps, n_features),filters=256,
                               kernel_size=4,
                               strides=1,
                               activation='tanh',
                               padding='same',
                               kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling1D())
model.add(Dropout(0.3))
model.add(Conv1D(input_shape=(n_steps, n_features),filters=128,
                               kernel_size=4,
                               strides=1,
                               activation='tanh',
                               padding='same',
                               kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling1D())
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dense(8, activation='softmax'))
model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics='acc')
history = model.fit(X, trainy, epochs=2000,validation_split=0.2,callbacks=callbacks_list,shuffle=False,verbose=1)   


show_history(history)
plot_history(history, path="standard.png")
plt.close()
model.save('E:/Biostatistics Master/BIOSTAT626/Midterm1/LSTM_grid/2Conv1d_BILSTM_256_128_64_9955.h5')

X, y = split_sequences(training, n_steps)
tX, ty = split_sequences(testing, n_steps)
    
trainy = to_categorical(y)

pre = []
for i in range(tX.shape[0]):
    yhat = model.predict(tX[i].reshape(1,n_steps,561))
    pre.append(yhat)

pre = np.array(pre)
test_class = np.argmax(pre.reshape(tX.shape[0],8), axis=1)
test_class = test_class.tolist()
test_class = pd.DataFrame(test_class)
test_class.to_csv('E:/Biostatistics Master/BIOSTAT626/Midterm1/2Conv1d_BILSTM_256_128_64_9955.csv')
