#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, ReLU, BatchNormalization
from tensorflow.keras.models import Model


def conv_block(inputs, base, batch_norm):
    
    #Two convolutional blocks as used in UNet.
    
    conv = Conv2D(filters=base, kernel_size=(3, 3), strides=(1, 1), padding= 'same') (inputs)
    
    if batch_norm:
        conv= BatchNormalization()(conv)
    x = ReLU()(conv)
        
    conv = Conv2D(filters=base, kernel_size=(3, 3), strides=(1, 1), padding= 'same') (x)
    
    if batch_norm:
        conv = BatchNormalization()(conv)
    out = ReLU()(conv)
                  
    return out


# In[2]:


def get_unet(img_h, img_w, img_c, base, batch_norm, dropout, dr = 0.2):
    input = Input ((img_h, img_w, img_c))

    # Encoder
    enc1 = conv_block(input, base, batch_norm)
    pool1 =MaxPooling2D (2, 2) (enc1) 
    if dropout:
        pool1=Dropout(dr)(pool1)

    enc2 = conv_block(pool1, base*2, batch_norm) 
    pool2 = MaxPooling2D (2, 2) (enc2) 
    if dropout:
        pool2 = Dropout (dr) (pool2)

    enc3 = conv_block(pool2, base*4, batch_norm)
    pool3 = MaxPooling2D (2, 2) (enc3) 
    if dropout:
        pool3 = Dropout (dr) (pool3)

    enc4 = conv_block (pool3, base*8, batch_norm) 
    pool4= MaxPooling2D (2, 2) (enc4)

    if dropout:
        pool4 = Dropout (dr) (pool4)

    # Bottleneck
    bottle = conv_block (pool4, base*16, batch_norm) 
    
    # Decoder
    dec1 = Conv2DTranspose(base*8, (2, 2), strides=(2, 2), padding='same')(bottle)
    dec1 = concatenate([dec1, enc4])  # Corrected the variable name to "dec1" from "decl"
    if dropout:
        dec1 = Dropout(dr)(dec1)

    dec1 = conv_block(dec1, base*8, batch_norm)

    dec2 = Conv2DTranspose(base*4, (2, 2), strides=(2, 2), padding='same')(dec1)  # Corrected "strides=" from "strides-"
    dec2 = concatenate([dec2, enc3])

    if dropout:
        dec2 = Dropout(dr)(dec2)

    dec2 = conv_block(dec2, base*4, batch_norm) # Adding another convolution block as per your instructions

    dec3 = Conv2DTranspose(base*2, (2, 2), strides=(2, 2), padding='same')(dec2)
    dec3 = concatenate([dec3, enc2])

    if dropout:
        dec3 = Dropout(dr)(dec3)

    dec3 = conv_block(dec3, base*2, batch_norm)

    dec4 = Conv2DTranspose(base, (2, 2), strides=(2, 2), padding='same')(dec3)
    dec4 = concatenate([dec4, enc1])

    if dropout:
        dec4 = Dropout(dr)(dec4)

    dec4 = conv_block(dec4, base, batch_norm)

    # Final output layer for segmentation
    output = Conv2D(1, (1, 1), activation='sigmoid')(dec4)
    
    clf = tf.keras.Model(inputs=input, outputs=output)
    clf.summary()
    return clf

def get_multi_class_unet(img_h, img_w, img_c, num_classes, base, batch_norm, dropout, dr=0.2):
    inputs = Input((img_h, img_w, img_c))

    # Encoder
    enc1 = conv_block(inputs, base, batch_norm)
    pool1 = MaxPooling2D((2, 2))(enc1)
    if dropout:
        pool1 = Dropout(dr)(pool1)

    enc2 = conv_block(pool1, base * 2, batch_norm)
    pool2 = MaxPooling2D((2, 2))(enc2)
    if dropout:
        pool2 = Dropout(dr)(pool2)

    enc3 = conv_block(pool2, base * 4, batch_norm)
    pool3 = MaxPooling2D((2, 2))(enc3)
    if dropout:
        pool3 = Dropout(dr)(pool3)

    enc4 = conv_block(pool3, base * 8, batch_norm)
    pool4 = MaxPooling2D((2, 2))(enc4)

    if dropout:
        pool4 = Dropout(dr)(pool4)

    # Bottleneck
    bottle = conv_block(pool4, base * 16, batch_norm)

    # Decoder
    dec1 = Conv2DTranspose(base * 8, (2, 2), strides=(2, 2), padding='same')(bottle)
    dec1 = concatenate([dec1, enc4])
    if dropout:
        dec1 = Dropout(dr)(dec1)

    dec1 = conv_block(dec1, base * 8, batch_norm)

    dec2 = Conv2DTranspose(base * 4, (2, 2), strides=(2, 2), padding='same')(dec1)
    dec2 = concatenate([dec2, enc3])

    if dropout:
        dec2 = Dropout(dr)(dec2)

    dec2 = conv_block(dec2, base * 4, batch_norm)

    dec3 = Conv2DTranspose(base * 2, (2, 2), strides=(2, 2), padding='same')(dec2)
    dec3 = concatenate([dec3, enc2])

    if dropout:
        dec3 = Dropout(dr)(dec3)

    dec3 = conv_block(dec3, base * 2, batch_norm)

    dec4 = Conv2DTranspose(base, (2, 2), strides=(2, 2), padding='same')(dec3)
    dec4 = concatenate([dec4, enc1])

    if dropout:
        dec4 = Dropout(dr)(dec4)

    dec4 = conv_block(dec4, base, batch_norm)

    # Final output layer for multi-class classification
    output = Conv2D(num_classes, (1, 1), activation='softmax')(dec4)

    model = Model(inputs=inputs, outputs=output)
    model.summary()
    return model


