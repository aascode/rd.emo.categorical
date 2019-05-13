# Categorized Speech Emotion Recognition
Implementation of several speech emotion recognition models.
Tested on computation server d8 (GPU: GTX980Ti * 4).

## Setup

### Database
[The Interactive Emotional Dyadic Motion Capture (IEMOCAP) Database](https://sail.usc.edu/iemocap/)

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

## How to run
See wiki for details of features and models.

### LLD-SVM-DNN
SVM or DNN models with LLD features
- LLD features extraction
  0. Set the paths of opensmile and IEMOCAP database in SmileFeatureExtraction.py.
  1. Run SmileFeatureExtraction.py to extract LLD features. The features will be saved in "./features/".
- Training models
  2. Run LLD-SVM-Train.py to train the SVM model. The models should be saved in "./models/SVM/" and the results should be saved in "./results/SVM/"
