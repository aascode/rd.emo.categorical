#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:35:45 2019

@author: zhuzhi
"""
import numpy as np
import pandas as pd
import keras.backend as K
from utils import preprocess
from utils import waua, plot_wauacm
from keras.models import load_model
from sklearn.metrics import confusion_matrix


def main():
    batch_size = 128
    configs = ["IS09", "IS10", "IS11", "IS12", "ComParE", "GeMAPS", "eGeMAPS"]
    # IEMOCAP imp NHSA
    emotionsTest = ["Neutral", "Happiness", "Sadness", "Anger"]
    database = "IEMOCAP"
    for config in configs:
        # Load features
        features = pd.read_csv("features/{}_{}.csv".format(database, config))
        features.set_index(["speaker", "emotion", "actType", "name"],
                           inplace=True)
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
            model = load_model("models/DNN/{}_{}.h5".format(config, sp))
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
            del model
            K.clear_session()
            print("speaker: " + str(sp))
            print(cmV)
            print(cmT)
        waV, uaV = np.around(waua(cmV), decimals=4)
        waT, uaT = np.around(waua(cmT), decimals=4)
        cmpV = cmV / np.reshape(np.sum(cmV, 1), (4, 1))
        cmpT = cmT / np.reshape(np.sum(cmT, 1), (4, 1))
        imageName = "results/DNN/Test_{}_V.png".format(config)
        title = "wa={}, ua={}".format(waV, uaV)
        plot_wauacm(title, cmpV, emotionsTest, imageName)
        imageName = "results/DNN/Test_{}_T.png".format(config)
        title = "wa={}, ua={}".format(waT, uaT)
        plot_wauacm(title, cmpT, emotionsTest, imageName)
        print("waV: " + str(waV) + ", uaV: " + str(uaV))
        print(cmpV)
        print("waT: " + str(waT) + ", uaT: " + str(uaT))
        print(cmpT)


if __name__ == "__main__":
    main()
