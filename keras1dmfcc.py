#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:12:37 2017

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


random.seed(a=1)
np.random.seed(2)
def get_audio_files(path, extension='wav'):
    files={}
    for root, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            for root2, dirnames2, filenames2 in os.walk(path+dirname):
                for filename3 in fnmatch.filter(filenames2, '*.'+extension):
                    files[path+dirname+"/"+filename3]=dirname
    return files

def make_category(files):
    cats={}
    
    speakers=list(set(files.values()))
    nclasses=len(speakers)
    onehotmat=np.eye(nclasses)
    for i in range(nclasses):
        cats[speakers[i]]=onehotmat[i,:]
    return cats

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

def add_noise(X,Y,times):
    
    for root, dirnames, filenames in os.walk(pathnoise):
        l=len(X)
        for i in range(times):
            global aud,noise
            aud=X[i%l]
            random.shuffle(filenames)
            for fn in filenames:
                #print root+fn
                noise,s=librosa.load(root+fn)
                break
            scale=np.random.uniform(low=0.1,high=0.2,size=1)
            mix=(aud[0:80000] + scale*noise[0:80000])/(1+scale)
            mfcc=get_features(mix,s)
            X.append(mfcc)
            if i%100==0:
                print "Adding noise:" + str(i)
            Y.append(Y[i%l])
        combined = list(zip(X, Y))
        random.shuffle(combined)

        X[:], Y[:] = zip(*combined)
        del combined
        return X,Y

def load_files(files,cats):
    X=[]
    Y=[]
    keys=files.keys()
    random.shuffle(keys)
    for i,key in enumerate(keys):
        aud,s=librosa.load(key)
        mfcc=get_features(aud,s)
        X.append(mfcc)
        speaker=files[key]
        Y.append(cats[speaker])
        #print i
        '''
        if i>10:
            break
            print i'''
    return X,Y

num_classes=5
nb_filter1=16
nb_filter2=8
nb_filter3=4
filter_length=3
filter_length2=2
length=39
model=Sequential()

model.add(Conv1D(filters=nb_filter1, kernel_size=filter_length, activation='relu', input_shape=(length, 1)))
#model.add(MaxPooling1D(pool_size=1, strides=None, padding='valid'))
model.add(Dropout(0.2))

model.add(Conv1D(filters=nb_filter2, kernel_size=filter_length2, activation='relu'))
#model.add(MaxPooling1D(pool_size=1, strides=None, padding='valid'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu',kernel_initializer=glorot_normal(2), kernel_constraint=None))

model.add(Dropout(0.5)) # CHECK IF NEEDED????

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae'])

model.summary()

load_from_dump=1
path='/media/alekh/01D1725AEE32E130/voxceleb/Train/'
pathnoise='/media/alekh/01D1725AEE32E130/voxceleb/noises/'
if load_from_dump!=1:
    print "Loading each file... "
    files=get_audio_files(path)
    cats=make_category(files)
    Xl,Yl=load_files(files,cats)
    #Xl,Yl=add_noise(Xl,Yl,4000)
    with open('Xl', 'w') as outfile:
    		pickle.dump(Xl, outfile)
    with open('Yl', 'w') as outfile:
    		pickle.dump(Yl, outfile)
else:
    print "Loading from dump... "
    with open('Xl', 'rb') as outfile:
        Xl=pickle.load(outfile)
    with open('Yl', 'rb') as outfile:
        Yl=pickle.load(outfile)


Xl=np.atleast_3d(np.array(Xl))
Yl=np.array(Yl)

nb_samples=Xl.shape[0]
test_size = int((0.02 * nb_samples+1))

X_train, X_test, Y_train, Y_test = Xl[:-test_size], Xl[-test_size:], Yl[:-test_size], Yl[-test_size:]

model.fit(X_train, Y_train, nb_epoch=60, batch_size=20, validation_data=(X_test, Y_test))

pred = model.predict(X_test)
predonehot = np.zeros_like(pred)
predonehot[np.arange(len(pred)), pred.argmax(1)] = 1
predmax=pred.argmax(1)
testmax=Y_test.argmax(1)
cmp=np.stack((predmax.T,testmax.T), axis=1)
#print predonehot
#print Y_test
check=np.abs(cmp[:,0]-cmp[:,1])
cnts=Counter(check)
no_zer=cnts[0]
print "No of correct classification: " + str(no_zer)
print "Percentage: " + str(float(no_zer*100)/float(check.shape[0]))


model.save('model60epoch.h5')

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])










