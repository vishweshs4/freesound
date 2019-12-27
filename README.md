# [Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging/overview)

The premise of this repository is:
* to provide an introduction to working with audio in Deep Learning context
* to evaluate the performance of [high resolution spectrograms](https://github.com/earthspecies/spectral_hyperresolution) against commonly used methods of generating spectrograms

The repository contains everything you need to get started with the [Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging/overview). The dataset is both interesting and challenging. [General audio tagging with ensembling convolutional neural network and statistical features](https://arxiv.org/abs/1810.12832) features a good overview of the competition.

When I want to learn something, I like to make it hard for whatever I am testing to prove its value. Here I first [picked an architecture](https://www.kaggle.com/c/freesound-audio-tagging/discussion/62634) that was demonstrated to perform well on this dataset on the usual variety of spectrograms.

I then proceeded to find hyperparameters for training that would work best for training on Mel-spectrograms with db loudness scale (visually these seem to produce the richest representation from the non high res spectrograms). Having found the parameters that perform best (length of training, differential lrs, augmentations) I proceeded to train on high resolution spectrograms. Since the parameters were optimized for Mel db spectrograms, high resolution spectrograms would need to contain more useful information by a significant margin to perform better in this very challenging setting. Here are the results that I got:

### Standard spectrograms

| frequency axis | loudness scale | top 3 error rate |
| ---            | ---            | ---              |
|linear          |linear          |0.112             |
|linear          |db              |0.101             |
|log             |linear          |0.010             |
|log             |db              |**0.093**         |
|mel             |linear          |0.113             |
|mel             |db              |0.104             |

### High resolution spectrograms

| frequency axis | loudness scale     | q  | top 3 error rate |
| ---            | ---                | ---| ---              |
|log             |db                  |1   |0.098             |
|log             |db                  |2   |0.099             |
|log             |db                  |4   |0.094             |
|log             |db                  |8   |**0.093**         |
