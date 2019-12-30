# [Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging/overview)

The premise of this repository is:
* to provide an introduction to working with audio in Deep Learning context
* to evaluate the performance of [high resolution spectrograms](https://github.com/earthspecies/spectral_hyperresolution) against commonly used methods of generating spectrograms

The repository contains everything you need to get started with the [Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging/overview). The dataset is both interesting and challenging. [General audio tagging with ensembling convolutional neural network and statistical features](https://arxiv.org/abs/1810.12832) has a good overview of the competition.

I first [picked an architecture](https://www.kaggle.com/c/freesound-audio-tagging/discussion/62634) that was demonstrated to perform well on this dataset on the usual variety of spectrograms.

I then proceeded to find hyperparameters for training that would work best for training on Mel-spectrograms with db loudness scale (visually these seem to produce the richest representation from the standard resolution spectrograms). Having found the parameters that perform best (length of training, differential lrs) I proceeded to train on high resolution spectrograms. Training with hyperparameters optimized for Mel db spectrograms create a challenging environment for high resolution spectrograms to demonstate their value on this task.

Here are the results that I got:

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

### 3 channel images

I created 3 different versions of 3 channel images. One comprising linear, Mel and log frequency db loudness spectrograms, another one only with high resolution spectrograms with varying `q` (1, 4 and 8) and one featuring 2 channels of high resolution spectrograms (with `q` of 1 and 8) and a log frequency and db loudness standard resolution spectrogram. They attain an error rate of 0.113, 0.094 and 0.085 respectively.

## Summary

On this specific dataset, with the model that I went for and this specific training regime, high resolution spectrograms do not perform better than standard spectrograms. Of note is the improved performance of combining standard resolution and high resolution spectrograms - this approach offered a very significant performance gain. It would be interesting to revisit the hyperparameters and experiment with other optimizers to see if additional performance could be squeezed out of 3 channel combinations.

It is unclear whether this dataset was geared towards exploring the strenghts of high resolution spectrograms. It would be interesting to evaluate their performance on other datasets, specifically ones where the differences between classes are less pronounced.
