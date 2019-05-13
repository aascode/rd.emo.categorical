#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:40:12 2019

@author: zhuzhi
"""

import os
import socket
import subprocess
import pandas as pd
from glob import glob


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


def SmileExtract(smilePath, configPath, configs, database, dataDf):
    for config in configs:
        smileConfig = configPath + configs[config]
        featureFile = "features/{}_{}.csv".format(database, config)
        for _, row in dataDf.iterrows():
            cmd = ('{0} -C "{1}" -I "{2}" -csvoutput "{3}" -instname '
                   + '"{4}"').format(smilePath, smileConfig,
                                     row.soundPath, featureFile, row.soundPath)
            subprocess.Popen(cmd, shell=True)
            print("", end="")
        featureDf = pd.read_csv("features/{}_{}.csv".format(database, config),
                                sep=";")
        featureDf.drop("frameTime", axis=1, inplace=True)
        featureDf["emotion"] = dataDf.emotion
        featureDf["speaker"] = dataDf.speaker
        if "actType" in dataDf.columns:
            featureDf["actType"] = dataDf.actType
            featureDf.set_index(["actType", 'speaker', 'emotion', 'name'],
                                inplace=True)
        else:
            featureDf.set_index(['speaker', 'emotion', 'name'],
                                inplace=True)
        featureDf.sort_index(inplace=True)
        featureDf.to_csv(featureFile)


def main():
    # define paths of SMILExtract, configs of open simle, and IEMOCAP database
    if socket.getfqdn(socket.gethostname()) == "d8":
        smilePath = "/home/zhu/Library/opensmile/bin/SMILExtract"
        configPath = "/home/zhu/Library/opensmile-2.3.0/config/"
    else:
        smilePath = "/usr/local/bin/SMILExtract"
        configPath = "/Users/zhuzhi/Library/opensmile-2.3.0/config/"
    dataPath = "../../../Database/IEMOCAP_full_release/"
    # SIMLE configs
    configs = {"IS09": "IS09_emotion.conf",
               "IS10": "IS10_paraling.conf",
               "IS11": "IS11_speaker_state.conf",
               "IS12": "IS12_speaker_trait.conf",
               "ComParE": "IS13_ComParE.conf",
               "GeMAPS": "gemaps/GeMAPSv01a.conf",
               "eGeMAPS": "gemaps/eGeMAPSv01a.conf"}
    # emotions and acting types
    database = "IEMOCAP"
    emotionsTest = ["Neutral", "Happiness", "Sadness", "Anger", "Boredom",
                    "Disgust", "Fear", "Ecited", "Frustration", "Surprise"]
    actTypeToUse = ["impro", "script"]
    # extraction
    if not os.path.exists("features"):
        os.mkdir("features")
    dataDf = load_IEMOCAP(dataPath, actTypeToUse, emotionsTest)
    SmileExtract(smilePath, configPath, configs, database, dataDf)


if __name__ == "__main__":
    main()
