#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:23:03 2019

@author: zhuzhi
"""

import os
import utils
import numpy as np


def main():
    dataPath = "../../../Database/IEMOCAP_full_release/"
    # Spectrograms
    if not os.path.exists("spectrograms"):
        os.mkdir("spectrograms")
    # IEMOCAP
    actTypeToUse = ["impro"]
    emotionsTest = ["Neutral", "Happiness", "Sadness", "Anger"]
    dataDf = utils.load_IEMOCAP(dataPath, actTypeToUse, emotionsTest)
    # params for stft
    wsize = 0.04  # second
    woverlap = 0.75  # %
    # signal length
    xLens = []
    N = dataDf.shape[0]
    n = 0
    print("Calculating spectrogram...")
    for _, row in dataDf.iterrows():
        xfft = utils.spectrogram(row.soundPath, wsize, woverlap)
        xLen = xfft.shape[1]
        xLens.append(xLen)
        n += 1
        print("\r{}% finished.".format(round(n/N*100, 2)),
              end="")
        np.save("spectrograms/{}.npy".format(row.filename), xfft)
    dataDf["Length"] = xLens
    dataDf.to_csv("spectrograms/IEMOCAP.csv")
    print()


if __name__ == "__main__":
    main()
