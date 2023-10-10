#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize


def generate_lenet_model(img_ch, img_width, img_height, base, drop, d_rate, num_classes):
    inputs_layer = Input(shape=(img_width, img_height, img_ch))
    x = Conv2D(filters=base, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(inputs_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=base*2, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(base*2, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)

    clf = tf.keras.Model(inputs=inputs_layer, outputs=out)
    clf.summary()
    return clf

def generate_alexnet_model(img_ch, img_width, img_height, n_base,drop,d_rate, num_classes):
    
    inputs_layer = Input(shape=(img_width, img_height, img_ch))
    x = Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch), kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(inputs_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(x)
    x = Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(x)
    x = Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    if drop:
        x = Dropout(rate=d_rate)(x)
    x = Dense(64, activation='relu')(x)
    if drop:
        x = Dropout(rate=d_rate)(x)
    out = Dense(num_classes,activation='softmax')(x)
    
    clf = tf.keras.Model(inputs=inputs_layer, outputs=out)
    clf.summary()
    return clf

def vgg16_model(img_ch, img_width, img_height, base,drop,d_rate,num_classes):
    inputs_layer = Input(shape=(img_width, img_height, img_ch))
    x = Conv2D(filters=base, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(inputs_layer)
    x = Conv2D(filters=base, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=base*2, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = Conv2D(filters=base*2, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=base*4, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = Conv2D(filters=base*4, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = Conv2D(filters=base*4, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=base*8, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = Conv2D(filters=base*8, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = Conv2D(filters=base*8, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=base*8, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = Conv2D(filters=base*8, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = Conv2D(filters=base*8, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    if drop:
        x = Dropout(rate=d_rate)(x)
    x = Dense(64, activation='relu')(x)
    if drop:
        x = Dropout(rate=d_rate)(x)
    out = Dense(num_classes, activation='sigmoid')(x)
    
    clf = tf.keras.Model(inputs=inputs_layer, outputs=out)
    clf.summary()
    return clf

def compileandfit(net,Loss,opt,x_train,y_train,x_test,y_test,lr,batch,epochs,metrics):
    if opt == 'Adam':
        optimizer = Adam(lr=lr)
    elif opt == 'SGD':
        optimizer = SGD(lr=lr)
    else:
        optimizerr = RMSprop(lr=lr)
    
    net.compile(loss=Loss, optimizer = optimizer, metrics = metrics)
    model_hist = net.fit(x_train,y_train, validation_data=(x_test,y_test),batch_size=batch,epochs=epochs,verbose=2)
    return model_hist



get_ipython().run_line_magic('matplotlib', 'inline')


import matplotlib.pyplot as plt
def plotcurve(clf_hist,metrics):
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(clf_hist.history["loss"], label="training loss")
    plt.plot(clf_hist.history["val_loss"], label="validation loss")
    plt.plot( np.argmin(clf_hist.history["val_loss"]),
            np.min(clf_hist.history["val_loss"]),
            marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();

    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    if metrics == 'binary':
        plt.plot(clf_hist.history["binary_accuracy"], label="training accuracy")
        plt.plot(clf_hist.history["val_binary_accuracy"], label="validation accuracy")
        plt.plot( np.argmax(clf_hist.history["val_binary_accuracy"]),
                np.max(clf_hist.history["val_binary_accuracy"]),
                marker="x", color="r", label="best model")
    else:
        plt.plot(clf_hist.history["accuracy"], label="training accuracy")
        plt.plot(clf_hist.history["val_accuracy"], label="validation accuracy")
        plt.plot( np.argmax(clf_hist.history["val_accuracy"]),
                np.max(clf_hist.history["val_accuracy"]),
                marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Value")
    plt.legend();
    
    return plt


## Data Loader ##
import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
def gen_labels(im_name, classes):
    for idx, pat in enumerate(classes):
        if pat in im_name:
            return idx
    return len(classes)

classes = ['C1','C2','C3','C4','C5','C6','C7','C8','C9',]

def get_data(data_path, data_list, img_h, img_w):
    img_labels = []
    
    for item in enumerate(data_list):
         img = imread(os.path.join(data_path, item[1]), as_gray = True) # "as_grey"
         img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
         img_labels.append([np.array(img), gen_labels(item[1], classes)])

         if item[0] % 100 == 0:
             print('Reading: {0}/{1} of train images'.format(item[0], len(data_list)))
    shuffle(img_labels)
    return img_labels

def get_data_arrays(nested_list, img_h, img_w):
     
     img_arrays = np.zeros((len(nested_list), img_h, img_w), dtype = np.float32)
     label_arrays = np.zeros((len(nested_list)), dtype = np.int32)
     for ind in range(len(nested_list)):
         img_arrays[ind] = nested_list[ind][0]
         label_arrays[ind] = nested_list[ind][1]
     img_arrays = np.expand_dims(img_arrays, axis =3)
     return img_arrays, label_arrays

def get_train_test_arrays(train_data_path, test_data_path, train_list,
 test_list, img_h, img_w):

     train_data = get_data(train_data_path, train_list, img_h, img_w)
     test_data = get_data(test_data_path, test_list, img_h, img_w)

     train_img, train_label = get_data_arrays(train_data, img_h, img_w)
     test_img, test_label = get_data_arrays(test_data, img_h, img_w)
     del(train_data)
     del(test_data)
     return train_img, test_img, train_label, test_label


# In[ ]:




