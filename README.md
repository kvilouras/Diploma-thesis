# Diploma-thesis
Title: "Investigation of acoustic scene classification techniques"

## Versions used: 
- Keras 2.3.0 
- Tensorflow-gpu 2.1.0
- Librosa 0.7.1
- Pysoundfile 0.10.3



## Jupyter notebooks
- Run **_Training_stage.ipynb_** to load filenames and labels, pre-process your data (choose either Per-Channel Energy Normalization or log-Mel power spectrograms) and train your model. Multiple CNN architectures are available in the repository (see corresponding Python scripts). 

- Run **_MLP+ETi.ipynb_** for feature extraction and Enhanced Temporal Integration. An efficient MLP architecture is provided for training (can be trained on CPU, ~ 1 sec per epoch).

- Run **_grad-CAM.ipynb_** for 'Gradient-weighted Class Activation Mapping' visualization technique. Make sure you use the same pre-processing step throughout this process.


Also, keep an eye on the comments since they provide useful explanations for every step.
