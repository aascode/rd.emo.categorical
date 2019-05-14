#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:19:24 2019

@author: zhuzhi
"""
import utils
import keras
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix


def main():
    dataPath = "../../../Database/IEMOCAP_full_release/"
    # hyperparams
    tMax = 300
    fMax = 400
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
        model = load_model("models/CNN/{}.h5".format(sp_test))
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
        del model
        keras.backend.clear_session()
    # Evaluation
    waV, uaV = np.around(utils.waua(cmV), decimals=4)
    waT, uaT = np.around(utils.waua(cmT), decimals=4)
    cmpV = cmV / np.reshape(np.sum(cmV, 1), (4, 1))
    cmpT = cmT / np.reshape(np.sum(cmT, 1), (4, 1))
    imageName = "results/CNN/Test_{}_CNN_V.png".format(dataname)
    title = "wa={}, ua={}".format(waV, uaV)
    utils.plot_wauacm(title, cmpV, emotionsTest, imageName)
    imageName = "results/CNN/Test_{}_CNN_T.png".format(dataname)
    title = "wa={}, ua={}".format(waT, uaT)
    utils.plot_wauacm(title, cmpT, emotionsTest, imageName)
    print("waV: " + str(waV) + ", uaV: " + str(uaV))
    print(cmpV)
    print("waT: " + str(waT) + ", uaT: " + str(uaT))
    print(cmpT)


if __name__ == "__main__":
    main()
