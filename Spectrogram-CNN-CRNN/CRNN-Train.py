#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:14:19 2019

@author: zhuzhi
"""

import os
import utils
import keras
import socket
import numpy as np
from keras.models import Sequential
from sklearn.utils import class_weight
from keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPooling2D, Activation, Bidirectional
from keras.layers import Dense, Dropout, BatchNormalization, Reshape, CuDNNGRU


def crnn_model(num_classes, input_shape, dr, lr):
    model = Sequential()
    model.add(Conv2D(6,
                     kernel_size=(1, 12),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(10,
                     kernel_size=(1, 8)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Reshape((300, 930)))
    model.add(Bidirectional(CuDNNGRU(128, activation='relu')))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dr))
    model.add(Dense(num_classes, activation='softmax'))
    if socket.getfqdn(socket.gethostname()) == "d8":
        model = multi_gpu_model(model, gpus=2)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=lr, decay=0.0001),
                  metrics=['accuracy'])
    return model


def main():
    dataPath = "../../../Database/IEMOCAP_full_release/"
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/CRNN"):
        os.mkdir("results/CRNN")
    if not os.path.exists("models"):
        os.mkdir("models")
    if not os.path.exists("models/CRNN"):
        os.mkdir("models/CRNN")
    if socket.getfqdn(socket.gethostname()) == "d8":
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # hyperparams
    dr = 0.3
    lr = 0.006
    tMax = 300
    fMax = 400
    input_shape = (tMax, fMax, 1)
    batch_size = 64 * 2
    # IEMOCAP
    database = "IEMOCAP"
    emotionsTest = ["Neutral", "Happiness", "Sadness", "Anger"]
    num_classes = len(emotionsTest)
    actTypeToUse = ["impro"]
    emoTest = "".join([emo[0] for emo in actTypeToUse+emotionsTest])
    dataname = "{}_{}".format(database, emoTest)
    dataDf = utils.load_IEMOCAP(dataPath, actTypeToUse, emotionsTest)
    # confusion matrix
    cmV = np.zeros((num_classes, num_classes), dtype=int)
    cmT = np.zeros((num_classes, num_classes), dtype=int)
    for sp_test in range(10):
        print("test speaker:", sp_test)
        (x_train, y_train, y_train_raw,
         x_val, y_val, y_val_raw,
         x_test, y_test, y_test_raw) = utils.Preprocess(sp_test, emotionsTest,
                                                        dataDf, tMax, fMax)
        # set training parameters
        mc_cb = keras.callbacks.ModelCheckpoint('BestModel_CRNN.h5',
                                                monitor='val_acc',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True)
        mc_es = keras.callbacks.EarlyStopping(monitor="acc",
                                              patience=10,
                                              verbose=1)
        # class weight
        cw = class_weight.compute_class_weight('balanced',
                                               np.unique(y_train_raw[:, 0]),
                                               y_train_raw[:, 0])
        cw2 = {0: cw[0], 1: cw[1], 2: cw[2], 3: cw[3]}
        model = crnn_model(num_classes=num_classes,
                           input_shape=input_shape,
                           dr=dr, lr=lr)
        epochs = 16
        model.fit(x_train,
                  y_train,
                  class_weight=cw2,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_val, y_val))
        epochs = 50
        model.fit(x_train,
                  y_train,
                  class_weight=cw2,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[mc_cb, mc_es],
                  validation_data=(x_val, y_val))
        del mc_cb
        model.load_weights("BestModel_CRNN.h5")
        y_val_p = model.predict(x_val,
                                batch_size=batch_size,
                                verbose=1)
        y_val_p_C = utils.y_raw_transform(y_val_p, y_val_raw)
        cmV += confusion_matrix(y_val_raw[:, 0], y_val_p_C)
        y_test_p = model.predict(x_test,
                                 batch_size=batch_size,
                                 verbose=1)
        y_test_p_C = utils.y_raw_transform(y_test_p, y_test_raw)
        cmT += confusion_matrix(y_test_raw[:, 0], y_test_p_C)
        print("speaker: " + str(sp_test))
        print(cmV)
        print(cmT)
        model.save("models/CRNN/{}.h5".format(sp_test))
        del model
        keras.backend.clear_session()
    # Evaluation
    waV, uaV = np.around(utils.waua(cmV), decimals=4)
    waT, uaT = np.around(utils.waua(cmT), decimals=4)
    cmpV = cmV / np.reshape(np.sum(cmV, 1), (4, 1))
    cmpT = cmT / np.reshape(np.sum(cmT, 1), (4, 1))
    imageName = "results/CRNN/Train_{}_CRNN_V.png".format(dataname)
    title = "wa={}, ua={}".format(waV, uaV)
    utils.plot_wauacm(title, cmpV, emotionsTest, imageName)
    imageName = "results/CRNN/Train_{}_CRNN_T.png".format(dataname)
    title = "wa={}, ua={}".format(waT, uaT)
    utils.plot_wauacm(title, cmpT, emotionsTest, imageName)
    print("waV: " + str(waV) + ", uaV: " + str(uaV))
    print(cmpV)
    print("waT: " + str(waT) + ", uaT: " + str(uaT))
    print(cmpT)


if __name__ == "__main__":
    main()
