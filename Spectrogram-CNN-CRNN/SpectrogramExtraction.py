#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:22:38 2019

@author: zhuzhi
"""

import os
import numpy as np
from scipy import signal
import scipy.io.wavfile as wf
from utils import load_IEMOCAP


def spectrogram(x, fs, wsize, woverlap):
    _, _, xfft = signal.stft(x,
                             fs=fs,
                             window='hamming',
                             nperseg=wsize*fs,
                             noverlap=fs*wsize*woverlap,
                             nfft=1600)
    xfft = 20*np.log10(np.abs(xfft))  # to dB
    xfft = xfft[:400]  # only use 0~4 kHz bands
    return xfft


def main():
    dataPath = "../../../Database/IEMOCAP_full_release/"
    # Spectrograms
    if not os.path.exists("spectrogram"):
        os.mkdir("spectrogram")
    # IEMOCAP
    emotionsTest = ["Neutral", "Happiness", "Sadness", "Anger"]
    actTypeToUse = ["impro"]
    dataDf = load_IEMOCAP(dataPath, actTypeToUse, emotionsTest)
    # params for stft
    wsize = 0.04  # second
    woverlap = 0.75  # %
    # counting
    N = dataDf.shape[0]
    n = 0
    print("Calculating spectrogram...")
    for _, row in dataDf.iterrows():
        fs, x = wf.read(row.soundPath)
        xfft = spectrogram(x, fs, wsize, woverlap)
        np.save("spectrogram/{}.npy".format(row.filename), xfft)
        print("\r{}% finished.".format(round((n+1)/N*100, 2)), end="")
        n += 1
    print()


if __name__ == "__main__":
    main()
