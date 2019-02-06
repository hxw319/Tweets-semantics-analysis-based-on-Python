#coding:utf-8
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

N = 38
traindata = pd.read_csv('/Users/yueyue/Desktop/usc/M2S1/EE500/project/code/inputprocess-2/traindata_NN.csv')
trainlabel = pd.read_csv('/Users/yueyue/Desktop/usc/M2S1/EE500/project/code/inputprocess-2/trainlabel_NN.csv')
testdata = pd.read_csv('/Users/yueyue/Desktop/usc/M2S1/EE500/project/code/inputprocess-2/testdata_NN.csv')
testlabel = pd.read_csv('/Users/yueyue/Desktop/usc/M2S1/EE500/project/code/inputprocess-2/testlabel_NN.csv')

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = traindata.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

NN_model.fit(traindata, trainlabel, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

test_predictions = NN_model.predict(testdata)
mse = mean_squared_error(testlabel , test_predictions)
print(mse)

pred= pd.DataFrame(data=test_predictions)

pred.to_csv('/Users/yueyue/Desktop/usc/M2S1/EE500/project/code/inputprocess-2/NN/prediction.csv',header=0)
