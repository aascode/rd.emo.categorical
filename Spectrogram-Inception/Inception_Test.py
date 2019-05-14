#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:42:14 2019

@author: zhuzhi
"""
import os
import utils
import socket
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import load_model


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
        K.clear_session()
        # %% test
        model = load_model("models/Inception_{}.h5".format(sp_test))
        y_test_p = model.predict(x_test,
                                 batch_size=batch_size,
                                 verbose=1)
        cmTest += utils.y_raw_evaluate(y_test_p, y_test_raw)
        del model

    # %% print results
    waTest, uaTest = utils.waua(cmTest)
    # test set
    print("Results on test set: wa={}, ua={}".format(waTest, uaTest))
    print(cmTest)
    title = "wa={}, ua={}".format(waTest, uaTest)
    imageName = "results/Inception_Test_T.png"
    cmTestP = cmTest / np.reshape(np.sum(cmTest, 1), (4, 1))
    utils.plot_wauacm(title, cmTestP, emotionsTest, imageName, ".4f")


if __name__ == "__main__":
    main()
