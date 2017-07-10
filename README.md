# chainer v2 implementation of DenseNet-BC
This is an implementation of [DenseNet-BC](https://arxiv.org/abs/1608.06993).
The difference between DenseNet-BC and original DenseNet is using 1x1 convolutional layer before each 3x3 convolutional layer
in DenseBlock.

2 kinds of DenseNet is defined in `densenet.py`: `DenseNetCifar` & `DenseNetImagenet`.
Inputs for the former / latter are supposed to be 32x32 / 224x224 sized images.

# Results of cifar100, no tuning
## accuracy
![accuracy](https://raw.githubusercontent.com/crcrpar/ChainerDenseNet/plot/plot_images/accuracy.png)
## loss
![loss](https://raw.githubusercontent.com/crcrpar/ChainerDenseNet/plot/plot_images/loss.png)
