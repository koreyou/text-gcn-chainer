# Text GCN on Chainer

This project implements [Yao et al. 2018. Graph Convolutional Networks for Text Classification. ArXiv.](https://arxiv.org/abs/1809.05679) in [Chainer](https://chainer.org/).
The project includes code to reproduce text classification experiment on 20 news groups dataset. **This is NOT an official implementation by the authors.**

# How to Run

## Prerequisite

I have only tested the code on Python 3.6.4. Install dependent library by running:

```
pip install -r requirements.txt
```

You need to install `cupy` to enable GPU.

## Running training and evaluation

Run:

```
python train.py
```

Refer to `python train.py -h` for options.
Note that you can enable early stopping by `--early-stopping` flag, but overhead for saving intermediate is quite large.


# Reproducing the paper

This implementation reached test accuracy of 0.8671 in 20 News groups dataset.
Test accuracy was 0.8634 in the original paper, so the implementation has managed to reproduce the result in the paper.
It took 387 seconds on K80 GPU (Google Colaboratory).

I ran `python train.py -g 0` on git commit `49700d0` to achieve this result.
