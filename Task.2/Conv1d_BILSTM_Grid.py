# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 05:41:30 2023

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


N = [10,15,20]
net = [32,64,128]
kernel = [16,32,64]
L2 = [0.01,0.05,0]
n_features = 561

opt = tf.keras.optimizers.Adam(
    learning_rate=0.01)
def scheduler(epoch, lr=1e-2):
  if epoch < 100:
    return lr
  elif epoch < 250:
    return lr * tf.math.exp(-0.005)
  elif epoch < 500:
    return lr * tf.math.exp(-0.01)
  elif epoch < 1000:
    return lr * tf.math.exp(-0.02)
  else:
    return lr * tf.math.exp(-0.03)
checkpoint = ModelCheckpoint(filepath='E:/Biostatistics Master/BIOSTAT626/Midterm1//LSTM_grid/Conv1d_BILSTM.h5',
                                             monitor='val_acc',mode='auto' ,save_best_only='True')
    
callbacks_list = [
                     keras.callbacks.EarlyStopping(
                         monitor="val_acc",
                         patience=50,              
                     ),                      
                    keras.callbacks.LearningRateScheduler(scheduler),
                    checkpoint
]

for K in kernel:
    for nnlayers in net:
        for l2 in L2:
            for n_steps in N:
                X, y = split_sequences(training, n_steps)
                tX, ty = split_sequences(testing, n_steps)
    
                trainy = to_categorical(y)
                
                
                model = Sequential()
                model.add(Conv1D(input_shape=(n_steps, n_features),filters=32,
                               kernel_size=K,
                               strides=1,
                               activation='relu',
                               padding='same'))
                model.add(MaxPooling1D())
                model.add(Dropout(0.1))
                model.add(LSTM(nnlayers, activation='tanh',kernel_regularizer=regularizers.l2(l2),recurrent_regularizer=regularizers.l2(l2)
                                                    ))
                model.add(Dropout(0.1))
                model.add(Dense(8, activation='softmax'))
                model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics='acc')
                history = model.fit(X, trainy, epochs=8000,validation_split=0.2,callbacks=callbacks_list,shuffle=False,verbose=1)       
                
                model.load_weights('E:/Biostatistics Master/BIOSTAT626/Midterm1/LSTM_grid/Conv1d_BILSTM.h5')
                Best = round(max(history.history['val_acc']),4)
                
                model.save('E:/Biostatistics Master/BIOSTAT626/Midterm1/LSTM_grid/Conv/Conv1d_BILSTM_{}_{}_{}_{}_{}.h5'.format(n_steps,nnlayers,K,int(1000*l2),Best))
                print('The Conv1d_BILSTM with {} {} {} {} has the {}'.format(n_steps,nnlayers,K,l2,Best))