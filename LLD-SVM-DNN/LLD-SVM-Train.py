#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:47:20 2019

@author: zhuzhi
"""
import os
import pickle
import numpy as np
import pandas as pd
from utils import preprocess
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from utils import waua, plot_wauacm


def SVM_Train(config, emotionsTest, features, params):
    C, gamma = params
    cm = np.zeros((len(emotionsTest), len(emotionsTest)), dtype=int)
    nSpeaker = len(features.index.unique('speaker'))
    sps_val = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]
    for sp in range(nSpeaker):
        print("Training with test speaker {}".format(sp))
        (x_train_scaled, y_train,
         x_val_scaled, y_val,
         x_test_scaled, y_test) = preprocess(sp, sps_val[sp],
                                             features, emotionsTest)
        svm = SVC(class_weight="balanced", C=C, gamma=gamma)
        svm.fit(x_train_scaled, y_train)
        y_p = svm.predict(x_test_scaled)
        cm += confusion_matrix(y_test, y_p)
        modelfile = "models/SVM/{}_{}.sav".format(config, sp)
        pickle.dump(svm, open(modelfile, "wb"))
    wa, ua = waua(cm)
    cmp = cm / np.reshape(np.sum(cm, 1), (4, 1))
    imageName = "results/SVM/Train_{}.png".format(config)
    wa, ua = np.around(waua(cm), decimals=4)
    title = "C={}, gamma={}, wa={}, ua={}".format(C, gamma, wa, ua)
    plot_wauacm(title, cmp, emotionsTest, imageName)
    print("wa: " + str(wa) + ", ua: " + str(ua))
    print(cmp)


def main():
    if not os.path.exists("results/"):
        os.mkdir("results/")
    if not os.path.exists("models/"):
        os.mkdir("models/")
    if not os.path.exists("models/SVM"):
        os.mkdir("models/SVM")
    if not os.path.exists("results/SVM"):
        os.mkdir("results/SVM")
    configs = ["IS09", "IS10", "IS11", "IS12", "ComParE", "GeMAPS", "eGeMAPS"]
    bestParams = {"IS09": [8, 0.125],
                  "IS10": [8, 0.03125],
                  "IS11": [8, 0.03125],
                  "IS12": [8, 0.0078125],
                  "ComParE": [8, 0.0078125],
                  "GeMAPS": [8192, 0.0078125],
                  "eGeMAPS": [512, 0.125]}
    # IEMOCAP imp NHSA
    database = "IEMOCAP"
    emotionsTest = ["Neutral", "Happiness", "Sadness", "Anger"]
    for config in configs:
        features = pd.read_csv("features/{}_{}.csv".format(database, config))
        features.set_index(["speaker", "emotion", "actType", "name"],
                           inplace=True)
        print("config: " + config)
        SVM_Train(config, emotionsTest, features, bestParams[config])


if __name__ == "__main__":
    main()
