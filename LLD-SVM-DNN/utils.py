#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:53:09 2019

@author: zhuzhi
"""
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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


def preprocess(sp_test, sp_val, features, emotionsTest):
    # split train and test set
    speakers = features.index.unique('speaker').to_list()
    speaker_test = speakers.pop(sp_test)
    speaker_val = speakers.pop(sp_val)
    speaker_train = speakers

    x_test = features.loc[speaker_test].values
    y_test = (features.loc[speaker_test]
                      .index
                      .get_level_values('emotion')
                      .to_list())
    y_test = np.array([emotionsTest.index(emo) for emo in y_test])

    x_val = features.loc[speaker_val].values
    y_val = (features.loc[speaker_val]
                     .index
                     .get_level_values('emotion')
                     .to_list())
    y_val = np.array([emotionsTest.index(emo) for emo in y_val])

    x_train = features.loc[speaker_train].values
    y_train = (features.loc[speaker_train]
                       .index
                       .get_level_values('emotion')
                       .to_list())
    y_train = np.array([emotionsTest.index(emo) for emo in y_train])

    # normalization
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_val_scaled = scaler.transform(x_val)
    del x_train, x_val, x_test
    return x_train_scaled, y_train, x_val_scaled, y_val, x_test_scaled, y_test


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
