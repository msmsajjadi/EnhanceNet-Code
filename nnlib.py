#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import os
import scipy.misc
import numpy as np
from math import floor
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

""" helper functions """
def loadimg(fn, scale=4):
    try:
        img = Image.open(fn).convert('RGB')
    except IOError:
        return None
    w, h = img.size
    img.crop((0, 0, floor(w/scale), floor(h/scale)))
    img = img.resize((w//scale, h//scale), Image.ANTIALIAS)
    return np.array(img)/255

def saveimg(img, filename):
    img = 255*np.copy(img)
    if len(np.shape(img)) > 2 and np.shape(img)[2] == 1:
        img = np.reshape(img, (np.shape(img)[0], np.shape(img)[1]))
    img = scipy.misc.toimage(img, cmin=0, cmax=255)
    scipy.misc.imsave(filename, img)

""" neural network layers """
def conv(h, n=64):
    h = tf.contrib.layers.convolution2d(h, n, kernel_size=3, stride=1,
                                        padding='SAME', activation_fn=None)
    return h

def relu(h):
    h = tf.nn.relu(h)
    return h

def upsample(h):
    h = tf.image.resize_images(h, [2*tf.shape(h)[1], 2*tf.shape(h)[2]],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return h

def resi(h, sublayers):
    htemp = h
    for layer in sublayers:
        h = layer[0](h, *layer[1:])
    h += htemp
    return h

def NN(name, layers):
    h = layers[0]
    with tf.variable_scope(name, reuse=False) as scope:
        for layer in layers[1:]:
            h = layer[0](h, *layer[1:])
    return h
