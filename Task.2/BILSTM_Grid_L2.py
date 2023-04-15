# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 05:39:30 2023

@author: dzr
"""
#################################################
##      Grid Search for BILSTM with L2         ##
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
net = [64,128,256]
drop = [0.1,0.5]
L2 = [0.001,0]
n_features = 561

def scheduler(epoch, lr=1e-2):
  if epoch < 50:
    return lr
  elif epoch < 150:
    return lr * tf.math.exp(-0.005)
  elif epoch < 300:
    return lr * tf.math.exp(-0.01)
  elif epoch < 500:
    return lr * tf.math.exp(-0.02)
  else:
    return lr * tf.math.exp(-0.03)
checkpoint = ModelCheckpoint(filepath='E:/Biostatistics Master/BIOSTAT626/Midterm1//LSTM_grid/BILSTM.h5',
                                             monitor='val_acc',mode='auto' ,save_best_only='True')
    
callbacks_list = [
                     keras.callbacks.EarlyStopping(
                         monitor="val_acc",
                         patience=50,              
                     ),                      
                    keras.callbacks.LearningRateScheduler(scheduler),
                    checkpoint
]

##opt = optimizer=tfa.optimizers.RectifiedAdam(0.001) # sgd

for Drop in drop:
    for nnlayers in net:
        for l2 in L2:
            for n_steps in N:
                X, y = split_sequences(training, n_steps)
                tX, ty = split_sequences(testing, n_steps)
    
                trainy = to_categorical(y)
  
                model = Sequential()
                model.add(Bidirectional(LSTM(nnlayers, activation='tanh',kernel_regularizer=regularizers.l2(l2),recurrent_regularizer=regularizers.l2(l2)),
                                                    input_shape=(n_steps, n_features)))
                model.add(Dropout(Drop))
                model.add(Dense(8, activation='softmax'))
                model.compile(optimizer= 'sgd', loss=tf.keras.losses.CategoricalCrossentropy(), metrics='acc')
                history = model.fit(X, trainy, epochs=1500,validation_split=0.2,callbacks=callbacks_list,shuffle=False,verbose=1)       
                
                model.load_weights('E:/Biostatistics Master/BIOSTAT626/Midterm1/LSTM_grid/BILSTM.h5')
                Best = round(max(history.history['val_acc']),4)
                
                model.save('E:/Biostatistics Master/BIOSTAT626/Midterm1/LSTM_grid/Final/BILSTM_{}_{}_{}_{}_{}.h5'.format(n_steps,nnlayers,int(100*Drop),int(1000*l2),Best))
                print('The BILSTM with {} {} {} {} has the {}'.format(n_steps,nnlayers,Drop,l2,Best))

