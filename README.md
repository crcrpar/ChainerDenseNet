# chainer v2 implementation of DenseNet-BC
This is an implementation of [DenseNet-BC](https://arxiv.org/abs/1608.06993).
The difference between DenseNet-BC and original DenseNet is using 1x1 convolutional layer before each 3x3 convolutional layer
in DenseBlock.

3 kinds of DenseNet is defined in `densenet.py`: `DenseNetCifar`, `DenseNetImagenet`, & `DenseNet`.
`DenseNetCifar` and `DenseNetImagenet` consists of 4 DenseBlocks, while you can change # of blocks and # of layers for each block
by passing `n_layers` which is an argument of `DenseNet`.

## Environment
- Python 3.5.2
- Chainer 2.0.0
- ChainerCV 0.5.1
- OpenCV w/ contrib 3.2.0.7

**FYI**: You can install OpenCV with or without contrib for python via pip: `pip install opencv-python` or `pip install opencv-contrib-python`.
- [opencv-python](https://pypi.python.org/pypi/opencv-python)
- [opencv-contrib-python](https://pypi.python.org/pypi/opencv-contrib-python)

# Results of cifar100, no tuning
Whole training log is in [result_cifar100/log.json](https://github.com/crcrpar/ChainerDenseNet/blob/master/result_cifar100/log.json)
## accuracy
![accuracy](https://raw.githubusercontent.com/crcrpar/ChainerDenseNet/plot/plot_images/accuracy.png)
## loss
![loss](https://raw.githubusercontent.com/crcrpar/ChainerDenseNet/plot/plot_images/loss.png)

## Training
In my environment (GTX 1080), one epoch including validation took around 210 seconds.

## about model size
As to `DenseNetImagenet` or `DenseNet` in `densenet.py`, examples of # of layers are below.

| # of layers | growth rate | # of layer of each block |
|:-----------:|:-----------:|:------------------------:|
| 121         |     32      | 6, 12, 24, 16            |
| 169         |     32      | 6, 12, 32, 32            |
| 201         |     32      | 6, 12, 48, 32            |
| 161         |     48      | 6, 12, 36, 32            |
