# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:42:13 2019

@author: Acer
"""
import cv2
import numpy as np
import time
from PIL import Image
import math,os
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from Chinese_inputs import CommonChar, ImageChar
from test2 import Model


def build_generator(input_z,is_training, is_reuse):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x,name=name)
    with tf.variable_scope("generator",reuse=is_reuse) as scope:

        with tf.variable_scope("h0"):
            outputs = tf.layers.dense(input_z,512*4*4,kernel_initializer=w_init)
            outputs = tf.reshape(outputs,[-1,4,4,512])

            #outputs = tf.layers.batch_normalization(outputs, training=is_training, gamma_initializer=g_init)
            outputs = tf.nn.tanh(outputs)


        with tf.variable_scope("conv1"):
            outputs = tf.layers.conv2d_transpose(outputs,256,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)

            outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
            outputs = lrelu(outputs)


        with tf.variable_scope("conv2"):
            outputs = tf.layers.conv2d_transpose(outputs,128,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)

            outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
            outputs = lrelu(outputs)


        with tf.variable_scope("conv3"):
            outputs = tf.layers.conv2d_transpose(outputs,64,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
            outputs = lrelu(outputs)
        '''
        with tf.variable_scope("h0"):
            outputs = tf.layers.dense(input_z,256*8*8,kernel_initializer=w_init)
            outputs = tf.reshape(outputs,[-1,8,8,256])
            outputs = tf.layers.batch_normalization(outputs, training=is_training, gamma_initializer=g_init)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope("conv1"):
            outputs = tf.layers.conv2d_transpose(outputs,128,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            #outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
            outputs = tf.nn.relu(outputs)
        with tf.variable_scope("conv2"):
            outputs = tf.layers.conv2d_transpose(outputs,64,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            #outputs = tf.layers.batch_normalization(outputs,training=is_training,gamma_initializer=g_init)
            outputs = tf.nn.relu(outputs)
        '''
        with tf.variable_scope("outputs"):
            outputs = tf.layers.conv2d_transpose(outputs,1,(5,5),(2,2),'same',activation=None,kernel_initializer=w_init)
            outputs = tf.tanh(outputs)
    return outputs
batch_size=256

z =  np.random.normal(loc=0.0,scale=1.0,size=(256,90)) #np.random.uniform(-1,1,(model.batch_size,40))
c = np.random.multinomial(1,[0.1]*10,size=256)

T=101
sess = tf.Session()
while T-1:
    T=T-1
    if T==100:
        bool=False
    else:
        bool =True

    z =  np.random.normal(loc=0.0,scale=1.0,size=(256,90)) #np.random.uniform(-1,1,(model.batch_size,40))
    c = np.random.multinomial(1,[0.1]*10,size=256)
    input_z = tf.placeholder(tf.float32,shape=(batch_size,90))
    input_c = tf.placeholder(tf.float32,shape=(batch_size,10))
    input_zc = tf.concat((input_z,input_c),axis=-1)
    output=build_generator(input_zc,is_training=False,is_reuse=bool)
    tf.train.Saver().restore(sess,tf.train.latest_checkpoint('./ckpt/'))
    img = sess.run(output, feed_dict={input_z:z,input_c:c})
    print(T)

    '''
    sess=tf.Session()
    model=Model(256,sess)
    img=model.generate(z,c)
    '''

    for index, Img in enumerate(img[:100]):
        Img = Img * 127.5 + 127.5
        if len(Img.shape) == 3:
            Img = Img[:,:,0]
            Img = cv2.resize(Img, (128, 128))
            Image.fromarray(Img.astype(np.uint8)).save("./generate/" +str(100-T) + "_" +str(index)  + ".png")