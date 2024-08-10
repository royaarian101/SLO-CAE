# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 10:02:37 2023

designed Convolutional AutoEncoder for Feature Extraction of SLO images of eyes

@author: Roya Arian, royaarian101@gmail.com
"""


from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dropout, Conv2D
from keras.layers import concatenate, Conv2DTranspose, MaxPooling2D

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, name=None):
    """Function to add 3 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same', name=name)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # third layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same', name=name)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def CAE_feature_extractor(input_img, channel, n_filters = 32, dropout = 0.1, batchnorm = True):
    """Function to define the CAE Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)


    c5 = conv2d_block(p4, n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    p5 = MaxPooling2D((2, 2))(c5)
    p5 = Dropout(dropout)(p5)


    # Bottel neck (Middle)
    c6 = conv2d_block(p5, n_filters = n_filters * 32, kernel_size = 3, batchnorm = batchnorm, name='bottel_neck')

    ## encoder
    encoder = Model(input_img, c6)



    # Expansive Path

    u7 = Conv2DTranspose(n_filters * 16, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 16, kernel_size = 3, batchnorm = batchnorm)


    u8 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c4])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    u9 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    u10 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c9)
    u10 = concatenate([u10, c2])
    u10 = Dropout(dropout)(u10)
    c10 = conv2d_block(u10, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u11 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c10)
    u11 = Dropout(dropout)(u11)
    c11 = conv2d_block(u11, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)


    outputs = Conv2D(channel, (1, 1), activation='sigmoid')(c11)

    autoencoder = Model(inputs=[input_img], outputs=[outputs])

    return autoencoder, encoder