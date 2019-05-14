#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:53:09 2019

@author: zhuzhi
"""
import keras
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt


def load_IEMOCAP(dataPath, actTypeTest, emotionsTest):
    emotions = {"neu": "Neutral",
                "sad": "Sadness",
                "fea": "Fear",
                "xxx": "xxx",
                "hap": "Happiness",
                "exc": "Excited",
                "dis": "Disgust",
                "fru": "Frustration",
                "sur": "Surprise",
                "ang": "Anger",
                "oth": "Other"}
    dataList = []
    for nSes in range(1, 6):
        txtfiles = (glob('{}Session{}/dialog/EmoEvaluation/*.txt'
                         .format(dataPath, nSes)))
        for nEmoEva in range(len(txtfiles)):
            with open(txtfiles[nEmoEva]) as txtfile:
                for line in txtfile:
                    if line[0] == '[':
                        line = line.split()
                        filename = line[3]
                        filename_split = filename.split("_")
                        # informations about the sound file
                        soundPath = ("{}Session{}/sentences/wav/{}"
                                     .format(dataPath, nSes,
                                             filename_split[0]))
                        for m in range(1, len(filename_split)-1):
                            soundPath += "_" + filename_split[m]
                        soundPath += "/{}.wav".format(filename)
                        session = nSes
                        speaker = (filename_split[0][:5] + "_"
                                   + filename_split[-1][0])
                        if filename_split[1][0] == "i":
                            actingType = "impro"
                        else:
                            actingType = "script"
                        emotion = emotions[line[4]]
                        dataList.append([soundPath, filename, session, speaker,
                                         actingType, emotion])
    dataDf = pd.DataFrame(dataList, columns=["soundPath", "filename",
                                             "session", "speaker",
                                             "actType", "emotion"])
    dataDf.where(dataDf.actType.isin(actTypeTest), inplace=True)
    dataDf.where(dataDf.emotion.isin(emotionsTest), inplace=True)
    dataDf.dropna(inplace=True)
    dataDf.reset_index(drop=True, inplace=True)
    return dataDf


def waua(cm):
    # weighted & unweighted accuracy
    nEmotion = cm.shape[0]
    wa = np.sum(cm*np.eye(nEmotion))/np.sum(cm)
    ua = np.mean(np.sum(cm*np.eye(nEmotion), 1)/np.sum(cm, 1))
    return wa, ua


def plot_wauacm(title, cm, emotionsTest, imageName):
    # plot the results
    # cm = cm / np.reshape(np.sum(cm, 1), (4, 1))
    df_cm = pd.DataFrame(cm, index=emotionsTest, columns=emotionsTest)
    plt.figure(figsize=(6, 5))
    plt.title(title, fontsize='large', fontweight='bold')
    sns.heatmap(df_cm, annot=True, fmt=".04f")
    plt.savefig(imageName)
    plt.close()


def Split(data, tMax, fMax, emotionsTest, ss):
    print("Preprocessing {} set...".format(ss))
    x = np.zeros((0, tMax, fMax))
    y = np.zeros(0)
    y_raw = np.zeros((0, 2))  # (emotion, nSplit)
    for _, row in data.iterrows():
        # x
        xfft = np.load("spectrogram/{}.npy".format(row.filename)).T
        xLen = xfft.shape[0]
        nSplit = int(xLen/tMax) + 1
        SplitLen = int(xLen/nSplit)
        # y_raw
        emo = emotionsTest.index(row.emotion)
        y_raw = np.concatenate((y_raw, np.reshape(np.array([emo, nSplit]),
                                                  (1, 2))))
    nAllSplit = np.sum(y_raw[:, 1])
    x = np.zeros((int(nAllSplit), tMax, fMax))
    y = np.zeros(int(nAllSplit))
    nU = 0
    for _, row in data.iterrows():
        # x
        xfft = np.load("spectrogram/{}.npy".format(row.filename)).T
        xLen = xfft.shape[0]
        nSplit = int(xLen/tMax) + 1
        SplitLen = int(xLen/nSplit)
        # y_raw
        emo = emotionsTest.index(row.emotion)
        for ns in range(nSplit):
            xfft_split = np.concatenate((xfft[ns*SplitLen:(ns+1)*SplitLen],
                                         np.zeros((tMax-SplitLen, fMax))))
            x[nU] = xfft_split
            y[nU] = emo
            # print(speaker_test, row.filename, nU, x.shape, y.shape)
            nU += 1
            print("\r{}% finished.".format(round((nU)/nAllSplit*100, 2)),
                  end="")
    print()
    return x, y, y_raw


def Preprocess(sp_test, emotionsTest, dataDf, tMax, fMax):
    # Preprocess
    # split train, val, test set
    # val and test speaker should in the same session
    num_classes = len(emotionsTest)
    sps_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sps_val = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]
    speakers = list(dataDf.speaker.unique())
    speaker_test = speakers.pop(sps_test[sp_test])
    speaker_val = speakers.pop(sps_val[sp_test])
    speaker_train = speakers

    # Preprocess
    # create train data
    data_train = dataDf.where(dataDf.speaker.isin(speaker_train)).dropna()
    x_train, y_train, y_train_raw = Split(data_train, tMax, fMax,
                                          emotionsTest, "train")
    # create test data
    data_val = dataDf.where(dataDf.speaker == speaker_val).dropna()
    x_val, y_val, y_val_raw = Split(data_val, tMax, fMax,
                                    emotionsTest, "validation")
    # create test data
    data_test = dataDf.where(dataDf.speaker == speaker_test).dropna()
    x_test, y_test, y_test_raw = Split(data_test, tMax, fMax,
                                       emotionsTest, "test")
    # Normalization
    xM = np.mean(x_train)
    xStd = np.std(x_train)
    x_train = (x_train - xM) / xStd
    x_val = (x_val - xM) / xStd
    x_test = (x_test - xM) / xStd
    # for keras crnn model
    x_train = x_train.reshape(x_train.shape[0], tMax, fMax, 1)
    x_val = x_val.reshape(x_val.shape[0], tMax, fMax, 1)
    x_test = x_test.reshape(x_test.shape[0], tMax, fMax, 1)
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train, y_train_raw,
            x_val, y_val, y_val_raw,
            x_test, y_test, y_test_raw)


def y_raw_transform(y_p, y_raw):
    y_raw_p = np.zeros(0)
    nU = 0
    for y, nSplit in y_raw:
        logits = np.zeros(4)
        for nnU in range(int(nSplit)):
            logits += y_p[nU]
            nU += 1
        y_raw_p = np.concatenate((y_raw_p, np.reshape(np.argmax(logits), 1)))
    return y_raw_p
