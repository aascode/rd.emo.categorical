#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:12:58 2019

@author: zhuzhi
"""
import os
import keras
import socket
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import preprocess
from utils import waua, plot_wauacm
from keras.models import Sequential
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, BatchNormalization


def make_model(num_classes, input_shape, dr, lr):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dr))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dr))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dr))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['accuracy'])
    return model


def DNN_Train(features, emotionsTest, config):
    dr = 0.4
    lr = 0.0002
    batch_size = 128
    epochs = 100

    nSpeaker = len(features.index.unique('speaker'))
    cmV = np.zeros((len(emotionsTest), len(emotionsTest)), dtype=int)
    cmT = np.zeros((len(emotionsTest), len(emotionsTest)), dtype=int)
    for sp in range(nSpeaker):
        (x_train_scaled, y_train_C,
         x_val_scaled, y_val_C,
         x_test_scaled, y_test_C) = preprocess(sp, sp-1,
                                               features,
                                               emotionsTest)
        num_classes = len(emotionsTest)
        y_train = keras.utils.to_categorical(y_train_C, num_classes)
        # y_test = keras.utils.to_categorical(y_test_C, num_classes)
        y_val = keras.utils.to_categorical(y_val_C, num_classes)
        # DNN model
        input_shape = (features.shape[1],)
        mc_cb = keras.callbacks.ModelCheckpoint('BestModel_DNN.h5',
                                                monitor='val_acc',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True)
        es_cb = keras.callbacks.EarlyStopping(monitor="acc",
                                              patience=10,
                                              verbose=1)
        # class weight
        cw = class_weight.compute_class_weight('balanced',
                                               np.unique(y_train_C),
                                               y_train_C)
        cw2 = {0: cw[0], 1: cw[1], 2: cw[2], 3: cw[3]}
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default() as g:
            with tf.Session(graph=g):
                model = make_model(num_classes=num_classes,
                                   input_shape=input_shape,
                                   dr=dr, lr=lr)
#                if socket.getfqdn(socket.gethostname()) == "d8":
#                    model = multi_gpu_model(model, gpus=4)
                model.fit(x_train_scaled,
                          y_train,
                          class_weight=cw2,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          callbacks=[mc_cb, es_cb],
                          validation_data=(x_val_scaled, y_val))
                # evaluation
                del mc_cb
                model.load_weights('BestModel_DNN.h5')
                y_val_p = model.predict(x_val_scaled,
                                        batch_size=batch_size,
                                        verbose=1)
                y_val_p_C = [emoC.argmax() for emoC in y_val_p]
                cmV += confusion_matrix(y_val_C,
                                        y_val_p_C,
                                        labels=list(range(num_classes)))
                y_test_p = model.predict(x_test_scaled,
                                         batch_size=batch_size,
                                         verbose=1)
                cmT += confusion_matrix(y_test_C,
                                        [emoC.argmax() for emoC in y_test_p],
                                        labels=list(range(num_classes)))
                print("speaker: " + str(sp))
                print(cmV)
                print(cmT)
                model.save("models/DNN/{}_{}.h5".format(config, sp))
    waV, uaV = np.around(waua(cmV), decimals=4)
    waT, uaT = np.around(waua(cmT), decimals=4)
    cmpV = cmV / np.reshape(np.sum(cmV, 1), (4, 1))
    cmpT = cmT / np.reshape(np.sum(cmT, 1), (4, 1))
    imageName = "results/DNN/Train_{}_V.png".format(config)
    title = "wa={}, ua={}".format(waV, uaV)
    plot_wauacm(title, cmpV, emotionsTest, imageName)
    imageName = "results/DNN/Train_{}_T.png".format(config)
    title = "wa={}, ua={}".format(waT, uaT)
    plot_wauacm(title, cmpT, emotionsTest, imageName)
    print("waV: " + str(waV) + ", uaV: " + str(uaV))
    print(cmpV)
    print("waT: " + str(waT) + ", uaT: " + str(uaT))
    print(cmpT)


def main():
    if socket.getfqdn(socket.gethostname()) == "d8":
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    configs = ["IS09", "IS10", "IS11", "IS12", "ComParE", "GeMAPS", "eGeMAPS"]
    # IEMOCAP imp NHSA
    emotionsTest = ["Neutral", "Happiness", "Sadness", "Anger"]
    database = "IEMOCAP"
    if not os.path.exists("results/DNN"):
        os.mkdir("results/DNN")
    if not os.path.exists("models/DNN"):
        os.mkdir("models/DNN")
    for config in configs:
        # Load features
        features = pd.read_csv("features/{}_{}.csv".format(database, config))
        features.set_index(["speaker", "emotion", "actType", "name"],
                           inplace=True)
        # RandomSearchCV
        DNN_Train(features, emotionsTest, config)


if __name__ == "__main__":
    main()
