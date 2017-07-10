# chainer v2 implementation of DenseNet-BC
This is an implementation of DenseNet-BC.
The difference between DenseNet-BC and original DenseNet is using 1x1 convolutional layer before each 3x3 convolutional layer
in DenseBlock.

2 kinds of DenseNet is defined in `densenet.py`: `DenseNetCifar` & `DenseNetImagenet`.

