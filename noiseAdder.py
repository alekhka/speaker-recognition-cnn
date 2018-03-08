#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:07:03 2017

@author: alekh
"""

import librosa

pathaud='/media/alekh/01D1725AEE32E130/voxceleb/Train/Alan_Alda/1elTcNGC3q8_0000001.wav'
pathnoise='/home/alekh/Desktop/noises/Links from www_pacdv_com/airport-gate-1.mp3'

aud,s=librosa.load(pathaud)
noise,s2=librosa.load(pathnoise)

mix = (aud[0:80000]+0.2*noise[0:80000])/1.1

librosa.output.write_wav('mixed3',mix,s)
