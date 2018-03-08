#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:03:07 2017

@author: alekh
"""


import numpy as np
import os
import fnmatch
import random
import librosa
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import backend as K
from keras.initializers import glorot_normal
from collections import Counter
import cPickle as pickle
from keras.models import load_model
import matplotlib.pyplot as plt
from IPython.display import display

def get_features(y, sr):
    	y = y[0:80000] 	# analyze just 80k
    	#S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    	#S = librosa.feature.mfcc(y, sr=sr)
    	#log_S = librosa.logamplitude(S, ref_power=np.max)
    	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    	log_S = librosa.logamplitude(S, ref_power=np.max)
    	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    	delta_mfcc = librosa.feature.delta(mfcc)
    	delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    	#mean_mfcc = np.mean(S, 1)
    	#mean_mfcc = (mean_mfcc-np.mean(mean_mfcc))/np.std(mean_mfcc)
    	#var_mfcc = np.var(S, 1)
    	#var_mfcc = (var_mfcc-np.mean(var_mfcc))/np.std(var_mfcc)
    	#feature_vector = np.concatenate((mean_mfcc, var_mfcc))
    	feature_vector = np.concatenate((np.mean(mfcc,1), np.mean(delta_mfcc,1), np.mean(delta2_mfcc,1)))
    	feature_vector = (feature_vector-np.mean(feature_vector)) / np.std(feature_vector)
    	return feature_vector


model_path='/home/alekh/Desktop/Nokia/mfcc/model60epoch_91pc.h5'
model = load_model(model_path)
fns=[]
fns.append('/media/alekh/01D1725AEE32E130/voxceleb/test/0EuI2zU72uo_0000001.wav')
fns.append('/media/alekh/01D1725AEE32E130/voxceleb/test/0ApAnZOmR2I_0000001.wav')
fns.append('/media/alekh/01D1725AEE32E130/voxceleb/test/1elTcNGC3q8_0000001.wav')
fns.append('/media/alekh/01D1725AEE32E130/voxceleb/test/2z_hQp__dik_0000001.wav')
#fns.append('/media/alekh/01D1725AEE32E130/voxceleb/test/5HbYScltf1c_0000001.wav')
#fns.append('/home/alekh/Audio/2017-11-29-14:33:30.wav')
val=[2,1,4,5,'Lou','Alan']
#fns=['/media/alekh/01D1725AEE32E130/voxceleb/test/0EuI2zU72uo_0000001.wav','/media/alekh/01D1725AEE32E130/voxceleb/test/0ApAnZOmR2I_0000001.wav','/media/alekh/01D1725AEE32E130/voxceleb/test/1elTcNGC3q8_0000001.wav','/media/alekh/01D1725AEE32E130/voxceleb/test/2z_hQp__dik_0000001.wav','/media/alekh/01D1725AEE32E130/voxceleb/test/5HbYScltf1c_0000001.wav']
fns.append('/media/alekh/01D1725AEE32E130/voxceleb/Train/Louis_C.K./5HbYScltf1c_0000003.wav')
fns.append('/media/alekh/01D1725AEE32E130/voxceleb/Train/Alan_Alda/xHcXn2RLNS8_0000003.wav')

for i,filen in enumerate(fns):
    X,sr=librosa.load(filen)
    X=get_features(X[0:80000],sr)
    if i==3:
        z1= X[0:5]
    if i==4:
        z2= X[0:5]
    
    X=np.atleast_3d(np.array(X))
    st=time.time()
    pred=model.predict(X)
    print "The confidence scores for user " + str(val[i]) +" "+filen[44:]+ " is: "
    np.set_printoptions(suppress=True)
    print(pred[0])
    fig=plt.figure(i)
    plt.stem(pred[0])
    display(fig)
    plt.close()
    print " "
    #ka=0
    #for a in range(len(pred)):
    #    if pred[0][a]>0.3:
    #        ka=pred[a]
    #    print "The sample closely matches to "+str(a)+" speaker with "+str(pred[0][a])+" probability"
    en=time.time()
    #print en-st

print z1
print z2