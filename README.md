# chainer v2 implementation of DenseNet-BC
This is an implementation of [DenseNet-BC](https://arxiv.org/abs/1608.06993).
The difference between DenseNet-BC and original DenseNet is using 1x1 convolutional layer before each 3x3 convolutional layer
in DenseBlock.

2 kinds of DenseNet is defined in `densenet.py`: `DenseNetCifar` & `DenseNetImagenet`.
Inputs for the former / latter are supposed to be 32x32 / 224x224 sized images.

## Environment
- Python 3.5.2
- Chainer 2.0.0
- ChainerCV 0.5.1
- OpenCV w/ contrib 3.2.0.7

**FIY**: You can install OpenCV with or without contrib for python via pip: `pip install opencv-python` or `pip install opencv-contrib-python`.
- [opencv-python](https://pypi.python.org/pypi/opencv-python)
- [opencv-contrib-python](https://pypi.python.org/pypi/opencv-contrib-python)

# Results of cifar100, no tuning
## accuracy
![accuracy](https://raw.githubusercontent.com/crcrpar/ChainerDenseNet/plot/plot_images/accuracy.png)
## loss
![loss](https://raw.githubusercontent.com/crcrpar/ChainerDenseNet/plot/plot_images/loss.png)

## Training
In my environment (GTX 1080), one epoch including validation took around 210 seconds.
