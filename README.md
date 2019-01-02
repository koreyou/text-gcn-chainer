# Text GCN on Chainer

This project implements [Yao et al. 2018. Graph Convolutional Networks for Text Classification. ArXiv.](https://arxiv.org/abs/1809.05679) in [Chainer](https://chainer.org/).
The project includes codes to reproduce the text classification experiment on the 20 news groups dataset. **This is NOT an official implementation by the authors.**

This project adopts hyperparamters specified in the original paper.
I have greatly simplified text preprocessing to just splitting words with white spaces.

I referenced [@takyamamoto's implementation of GCN](https://github.com/takyamamoto/Graph-Convolution-Chainer) to implement this project.

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

Refer to `python train.py -h` for the options.
Note that you can enable early stopping by `--early-stopping` flag, but the overhead for saving intermediate models is quite large.


# Reproducing the paper

Running this project with the original adjacency matrix normalization method (`python train.py -g 0 --normalization gcn`) yields 0.8380 accuracy in the 20 News groups dataset.
The test accuracy was 0.8379 in the original paper.
The slighly inferior classification result may be due to simplified preprocessing.

Running `python train.py -g 0 --normalization pygcn` which uses normalization method proposed in [GCN authors' PyTorch implementation](https://github.com/tkipf/pygcn/issues/11) yields much better result of 0.8680 (comparable with the original paper).
