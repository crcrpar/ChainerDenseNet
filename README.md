# chainer v2 implementation of DenseNet-BC
This is an implementation of DenseNet-BC.
The difference between DenseNet-BC and original DenseNet is using 1x1 convolutional layer before each 3x3 convolutional layer
in DenseBlock.

2 kinds of DenseNet is defined in `densenet.py`: `DenseNetCifar` & `DenseNetImagenet`.
Inputs for the former / latter are 32x32 / 224x224 sized images.

# Results of cifar100, no tuning
![accuracy](https://raw.githubusercontent.com/crcrpar/ChainerDenseNet/plot/plot_images/accuracy.png)
![loss](https://github.com/crcrpar/ChainerDenseNet/blob/plot/plot_images/loss.png?raw=true)
