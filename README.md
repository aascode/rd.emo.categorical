# Categorized Speech Emotion Recognition
Implementation of several speech emotion recognition models.
Tested on computation server d8 (GPU: GTX980Ti * 4).

## Setup

### Database
[The Interactive Emotional Dyadic Motion Capture (IEMOCAP) Database](https://sail.usc.edu/iemocap/)
`share0/data/IEMOCAP/IEMOCAP_full_release_withoutVideos`

### Requirements
- opensmile 2.3.0
- python 3.5.2
- numpy 1.15.4
- scipy 1.2.1
- pandas 0.24.0
- matplotlib 3.0.2
- seaborn 0.9.0
- scikit-learn 0.20.2
- tensorflow-gpu 1.13.1
- keras 2.2.4
- librosa 0.6.3

## How to run
See [wiki](https://github.com/zhuzhi-fairy/rd.emo.categorical/wiki) for details of features and models.

### LLD-SVM-DNN
SVM and DNN models with LLD features
1. LLD features extraction
    - Set the paths of opensmile and IEMOCAP database in `SmileFeatureExtraction.py`.
    - Run `SmileFeatureExtraction.py` to extract LLD features. The features will be saved in "./features/".
2. Train models
    - Run `LLD-SVM-Train.py` to train the SVM model. The models should be saved in "./models/SVM/" and the results should be saved in "./results/SVM/"
    - Run `LLD-DNN-Train.py` to train the DNN model. The models should be saved in "./models/DNN/" and the results should be saved in "./results/DNN/"
3. Test models
    - Run `LLD-SVM-Test.py` to test the trained SVM model. The results should be saved in "./results/SVM/"
    - Run `LLD-DNN-Test.py` to test the trained DNN model. The results should be saved in "./results/DNN/"

### Spectrogram-CNN-CRNN
CNN and CRNN models with spectrogram as input
1. Spectrogram calculation
    - Set the paths of opensmile and IEMOCAP database in `SpectrogramExtraction.py`.
    - Run `SpectrogramExtraction.py` to calculate spectrograms. The spectrograms will be saved in "./spectrogram/".
2. Train models
    - Run `CNN-Train.py` to train the CNN model.
    - Run `CRNN-Train.py` to train the CRNN model.
3. Test models
    - Run `CNN-Test.py` to test the trained CNN model.
    - Run `CRNN-Test.py` to test the trained CRNN model.

### Spectrogram-Inception
An inception based CNN model with spectrogram as input
1. Spectrogram calculation
    - Set the paths of opensmile and IEMOCAP database in `SpectrogramExtraction.py`.
    - Run `SpectrogramExtraction.py` to calculate spectrograms. The spectrograms will be saved in "./spectrograms/".
2. Train model
    - Run `Inception-Train.py` to train the Inception model.
3. Test model
    - Run `Inception-Test.py` to test the trained Inception model.
