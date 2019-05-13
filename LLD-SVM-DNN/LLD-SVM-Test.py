#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:35:25 2019

@author: zhuzhi
"""
import pickle
import numpy as np
import pandas as pd
from utils import preprocess
from sklearn.metrics import confusion_matrix
from utils import waua, plot_wauacm


def main():
    configs = ["IS09", "IS10", "IS11", "IS12", "ComParE", "GeMAPS", "eGeMAPS"]
    database = "IEMOCAP"
    emotionsTest = ["Neutral", "Happiness", "Sadness", "Anger"]
    for config in configs:
        print("config: " + config)
        features = pd.read_csv("features/{}_{}.csv".format(database, config))
        features.set_index(["speaker", "emotion", "actType", "name"],
                           inplace=True)
        cm = np.zeros((len(emotionsTest), len(emotionsTest)), dtype=int)
        nSpeaker = len(features.index.unique('speaker'))
        sps_val = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]
        for sp in range(nSpeaker):
            print("Testing speaker {}".format(sp))
            (x_train_scaled, y_train,
             x_val_scaled, y_val,
             x_test_scaled, y_test) = preprocess(sp, sps_val[sp],
                                                 features, emotionsTest)
            modelfile = "models/SVM/{}_{}.sav".format(config, sp)
            svm = pickle.load(open(modelfile, "rb"))
            y_p = svm.predict(x_test_scaled)
            cm += confusion_matrix(y_test, y_p)
        wa, ua = waua(cm)
        cmp = cm / np.reshape(np.sum(cm, 1), (4, 1))
        imageName = "results/SVM/Test_{}.png".format(config)
        wa, ua = np.around(waua(cm), decimals=4)
        title = "wa={}, ua={}".format(wa, ua)
        plot_wauacm(title, cmp, emotionsTest, imageName)
        print("wa: " + str(wa) + ", ua: " + str(ua))
        print(cmp)


if __name__ == "__main__":
    main()
