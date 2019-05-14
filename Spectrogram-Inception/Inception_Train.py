#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:29:44 2019

@author: zhuzhi
"""

import os
import utils
import keras
import socket
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import keras.backend as K
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import concatenate, GlobalAveragePooling2D, Activation
from keras.utils import multi_gpu_model


def cnn_model(f_uplim, t_uplim, num_classes):
    x = Input(shape=(f_uplim, t_uplim, 1))
    # layer 1
    conv1f = Conv2D(8, (10, 2), padding="same")(x)
    conv1f_norm = BatchNormalization()(conv1f)
    conv1f_a = Activation("relu")(conv1f_norm)
    conv1t = Conv2D(8, (2, 8), padding="same")(x)
    conv1t_norm = BatchNormalization()(conv1t)
    conv1t_a = Activation("relu")(conv1t_norm)
    conv1 = concatenate([conv1f_a, conv1t_a], axis=-1)
    maxpooling1 = MaxPooling2D((2, 2))(conv1)
    # layer2
    conv2 = Conv2D(32, (3, 3))(maxpooling1)
    conv2_norm = BatchNormalization()(conv2)
    conv2_a = Activation("relu")(conv2_norm)
    maxpooling2 = MaxPooling2D((2, 2))(conv2_a)
    # layer3
    conv3 = Conv2D(48, (3, 3))(maxpooling2)
    conv3_norm = BatchNormalization()(conv3)
    conv3_a = Activation("relu")(conv3_norm)
    maxpooling3 = MaxPooling2D((2, 2))(conv3_a)
    # layer4
    conv4 = Conv2D(64, (3, 3))(maxpooling3)
    conv4_norm = BatchNormalization()(conv4)
    conv4_a = Activation("relu")(conv4_norm)
    # layer5
    conv5 = Conv2D(80, (3, 3))(conv4_a)
    conv5_norm = BatchNormalization()(conv5)
    conv5_a = Activation("relu")(conv5_norm)
    GAP = GlobalAveragePooling2D(data_format='channels_last')(conv5_a)
    y = Dense(num_classes, activation="softmax")(GAP)
    model = Model(inputs=x, outputs=y)
    if socket.getfqdn(socket.gethostname()) == "d8":
        model = model = multi_gpu_model(model, gpus=2)
    return model


def lrschedule(epochs, lr):
    if epochs == 1:
        return 0.05
    elif epochs == 20:
        return 0.005
    elif epochs == 30:
        return 0.0005
    elif epochs == 40:
        return 0.00005
    else:
        return lr


def main():
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("models"):
        os.mkdir("models")
    specPath = "spectrograms/"
    if socket.getfqdn(socket.gethostname()) == "d8":
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        batch_size = 64 * 2
    else:
        batch_size = 64
    # parameters
    t_uplim = 200  # 2 s
    f_uplim = 400  # 4 kHz
    t_shift_train = 100  # 1 s
    t_shift_test = 40  # 0.4 s

    # load the DataFrame of IEMOCAP
    dataDf = pd.read_csv(specPath + "IEMOCAP.csv", index_col=0)
    speakers = list(dataDf.speaker.unique())
    emotionsTest = list(dataDf.emotion.unique())
    num_classes = len(emotionsTest)

    # cross-validation
    cmTest = np.zeros((num_classes, num_classes))
    # %%
    for sp_test in range(len(speakers)):
        speakers_train = speakers.copy()
        speaker_test = speakers_train.pop(sp_test)
        print("Test speaker: {}".format(speaker_test))
        # preprocessing
        # train set
        (x_train, y_train, y_train_raw) = utils.prepro(specPath,
                                                       dataDf,
                                                       speakers_train,
                                                       emotionsTest,
                                                       "train",
                                                       t_shift_train,
                                                       f_uplim,
                                                       t_uplim)
        # test set
        (x_test, y_test, y_test_raw) = utils.prepro(specPath,
                                                    dataDf,
                                                    speaker_test,
                                                    emotionsTest,
                                                    "test",
                                                    t_shift_test,
                                                    f_uplim,
                                                    t_uplim)
        # class weight
        cw = class_weight.compute_class_weight('balanced',
                                               np.unique(y_train_raw[:, 0]),
                                               y_train_raw[:, 0])
        cw2 = {0: cw[0], 1: cw[1], 2: cw[2], 3: cw[3]}
        K.clear_session()
        model = cnn_model(f_uplim, t_uplim, num_classes)
        lrs_cb = keras.callbacks.LearningRateScheduler(lrschedule, verbose=1)
        epochs = 50
        model.compile(optimizer=SGD(lr=0.05,
                                    momentum=0.9,
                                    decay=0.0001,
                                    nesterov=True),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  class_weight=cw2,
                  callbacks=[lrs_cb],
                  validation_data=(x_test, y_test))
        # %% test
        y_test_p = model.predict(x_test,
                                 batch_size=batch_size,
                                 verbose=1)
        cmTest += utils.y_raw_evaluate(y_test_p, y_test_raw)
        model.save("models/Inception_{}.h5".format(sp_test))
        del model

    # %% print results
    waTest, uaTest = utils.waua(cmTest)
    # test set
    print("Results on test set: wa={}, ua={}".format(waTest, uaTest))
    print(cmTest)
    title = "wa={}, ua={}".format(waTest, uaTest)
    imageName = "results/Inception_Train_T.png"
    cmTestP = cmTest / np.reshape(np.sum(cmTest, 1), (4, 1))
    utils.plot_wauacm(title, cmTestP, emotionsTest, imageName, ".4f")


if __name__ == "__main__":
    main()
