# PixelCNN-Paddle

This is an unofficial Paddle implementation of [PixelCNN](https://arxiv.org/pdf/1601.06759v3.pdf) (Van Oord et al. 2016).

## Contents
1. [Introduction](#introduction)
2. [Reproduction Accuracy](#reproduction-accuracy)
3. [Dataset](#dataset)
4. [Environment](#environment)
5. [Train & Test](#train&test)
6. [Test](#test)
7. [Code Structure](#code-structure)

## Introduction

**Reference Code：**[PixelCNN](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/pixel_cnn.py)

**Paper：**[Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759v3.pdf)


## Reproduction Accuracy
In training, set batch size to 256.

| Index | Raw Paper| Reference Code | Reproduction |
| --- | --- | --- | --- |
| NLL| 81.30 | 85.74 | 86.00003680419921 |


## Dataset
The authors use MNIST dataset, and it will be auto-download when users training.


## Environment
- Frameworks: 
* [PaddlePaddle](https://paddlepaddle.org.cn/) (2.1.2)
* [NumPy](http://www.numpy.org/) (1.18.4)
* [Pillow](https://pillow.readthedocs.io/en/latest/index.html) (7.2.0)


## Train & Test

```
python train.py
```



## Code Structure

```
├── ckpts  # pdparams and training logs
├── src
│   ├── pixel_cnn.py
│   ├── datasets.py
│   ├── base.py
│   ├── convolution.py
│   ├── train.py
│   ├── trainer.py
├── README.md
└── requirements.txt
```
