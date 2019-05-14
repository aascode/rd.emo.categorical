#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:21:42 2019

@author: zhuzhi
"""
import keras
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
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


def prepro(specPath, dataDf, speakers, emotionsTest,
           ss, t_shift, f_uplim, t_uplim):
    if ss == "train":
        df = dataDf.where(dataDf.speaker.isin(speakers)).dropna()
    elif ss == "test":
        df = dataDf.where(dataDf.speaker == speakers).dropna()
    print("Preprocessing {} set...".format(ss))
    nUttrances = df.shape[0]
    nSplit_all = int(np.sum(np.ceil(np.array(df.Length)/t_shift)))
    x = np.zeros((nSplit_all, f_uplim, t_uplim))
    y = np.zeros(nSplit_all)
    y_raw = np.zeros((nUttrances, 2))  # (emotion, nSplit)

    nSub = 0
    for nU in range(nUttrances):
        row = df.iloc[nU]
        emo = emotionsTest.index(row.emotion)
        xfft = np.load("{}{}.npy".format(specPath, row.filename))
        xLen = int(row.Length)
        nSplit = int(xLen/t_shift) + 1
        y_raw[nU] = np.reshape(np.array([emo, nSplit]), (1, 2))
        pStart = 0
        while pStart < xLen:
            pEnd = pStart + t_uplim
            if pEnd > xLen:
                xfft_split = np.hstack((xfft[:, pStart:xLen],
                                        np.zeros((f_uplim, pEnd-xLen))))
            else:
                xfft_split = xfft[:, pStart:pEnd]
            # normalization
            scaler = MinMaxScaler(feature_range=(-1, 1))
            xfft_split_temp = scaler.fit_transform(xfft_split)
            xfft_split_a = np.log(1+255*np.abs(xfft_split_temp))/np.log(1+255)
            xfft_split = np.sign(xfft_split_temp) * xfft_split_a
            x[nSub] = np.reshape(xfft_split, (1, f_uplim, t_uplim))
            y[nSub] = np.array([emo])
            pStart += t_shift
            print("\r{}% finished.".format(round((nSub+1)/nSplit_all*100, 2)),
                  end="")
            nSub += 1
    print()
    # for keras cnn model
    x = x.reshape(x.shape[0], f_uplim, t_uplim, 1)
    x = x.astype('float32')
    y = keras.utils.to_categorical(y, len(emotionsTest))
    return x, y, y_raw


def y_raw_evaluate(y, y_raw):
    nUttrances = y_raw.shape[0]
    y_raw_p = np.zeros(nUttrances)
    nsub = 0
    for nU in range(nUttrances):
        num_subs = int(y_raw[nU, 1])
        y_raw_p[nU] = np.argmax(np.mean(y[nsub:nsub+num_subs], axis=0))
        nsub += num_subs
    cm = confusion_matrix(y_raw[:, 0], y_raw_p)
    return cm


def waua(cm):
    num_classes = cm.shape[0]
    wa = np.around(np.sum(cm*np.eye(num_classes))/np.sum(cm), decimals=4)
    ua = np.around(np.mean(np.sum(cm*np.eye(num_classes), 1)/np.sum(cm, 1)),
                   decimals=4)
    return wa, ua


def plot_wauacm(title, cm, emotionsTest, imageName, fmt):
    # plot the results
    # cm = cm / np.reshape(np.sum(cm, 1), (4, 1))
    df_cm = pd.DataFrame(cm, index=emotionsTest, columns=emotionsTest)
    plt.figure(figsize=(6, 5))
    plt.title(title, fontsize='large', fontweight='bold')
    sns.heatmap(df_cm, annot=True, fmt=fmt)
    plt.savefig(imageName)
    plt.close()
