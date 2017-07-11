# coding: utf-8
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers


class DenseLayer(chainer.Chain):

    """element of DenseBlock"""

    def __init__(self, in_ch, growth_rate, bn_size, dropout_rate=0.5):
        r"""initialization of denselayer.

        Args:
            in_ch (int): the number of inputs' channels
            growth_rate (int): growth_rate
            bn_size (int): bn size
            dropout_rate (float): dropout rate \in [0, 1). If > 0,
                apply dropout
        """
        super(DenseLayer, self).__init__()
        with self.init_scope():
            initialW = initializers.HeNormal()
            self.bn_1 = L.BatchNormalization(in_ch)
            self.conv_1 = L.Convolution2D(
                None, bn_size * growth_rate, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn_2 = L.BatchNormalization(bn_size * growth_rate)
            self.conv_2 = L.Convolution2D(
                None, growth_rate, 3, 1, 1, initialW=initialW, nobias=True)
        self.dropout_rate = dropout_rate
        self.in_ch = in_ch

    def __call__(self, x):
        """Feed-Forward calculation."""
        h = self.conv_1(F.relu(self.bn_1(x)))
        h = self.conv_2(F.relu(self.bn_2(h)))
        if self.dropout_rate > 0:
            h = F.dropout(h, self.dropout_rate)
        return F.concat((x, h))


class DenseBlock(chainer.Chain):

    """One block of DenseNet."""

    def __init__(self, n_layers, in_ch, bn_size, growth_rate, dropout_rate=0.5):
        r"""initialization of Dense Block.

        Args:
            n_layers (int): # of DenseLayer in DenseBlock
            in_ch (int): # of input tensor's channel
            bn_size (int): default, 4.
            growth_rate (int):
            dropout_rate (float, \in [0, 1)): if greater than 0, use dropout_rate
        """
        super(DenseBlock, self).__init__()
        with self.init_scope():
            for i in range(n_layers):
                tmp_in_ch = in_ch + i * growth_rate
                setattr(self, 'denselayer{}'.format(
                    i + 1), DenseLayer(tmp_in_ch, growth_rate, bn_size, dropout_rate))
        self.n_layers = n_layers

    def __call__(self, x):
        """Feed-Forward calculation."""
        h = x
        for i in range(1, self.n_layers + 1):
            h = getattr(self, 'denselayer{}'.format(i))(h)
        return h


class Transition(chainer.Chain):

    """Transition unit.

    This block halve width and height of inputs.
    """

    def __init__(self, in_ch, dropout_rate):
        r"""Initialization of Transition.

        Args:
            in_ch (int): # of inputs' channels
            dropout_rate (float): dropout_rate \in (0, 1]. If > 0,
                apply dropout
        """
        super(Transition, self).__init__()
        with self.init_scope():
            initialW = initializers.HeNormal()
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(
                in_ch, in_ch, 1, initialW=initialW, nobias=True)
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        """Feed-Forward calculation."""
        h = F.relu(self.bn(x))
        h = F.dropout(self.conv(h), self.dropout_rate)
        h = F.average_pooling_2d(h, 2)
        return h


class DenseNetCifar(chainer.Chain):

    """DenseNet for CIFAR10 / CIFAR100.

    This class takes as input (n_channel, 32, 32) sized images.
    """

    def __init__(self, growth_rate=32, n_layers=(6, 12, 24, 16), init_features=64, bn_size=4, dropout_rate=0, n_class=1000):
        r"""Initialization of DenseNetImagenet.

        Args:
            growth_rate (int): growth_rate
            n_layers (int): # of layers for each block
            init_features (int): # of self.conv1's outputs' channels
            bn_size (int): bn_size
            dropout_rate (float): dropout rate \in [0, 1). If > 0,
                apply dropout
            n_class (int): # of class
        """
        super(DenseNetCifar, self).__init__()
        with self.init_scope():
            initialW = initializers.HeNormal()
            self.conv1 = L.Convolution2D(
                None, init_features, 7, 2, 3, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(init_features)

            self.block1 = DenseBlock(
                n_layers[0], init_features, bn_size, growth_rate, dropout_rate)
            n_feature = init_features + n_layers[0] * growth_rate
            self.trans1 = Transition(n_feature, dropout_rate)

            self.block2 = DenseBlock(
                n_layers[1], n_feature, bn_size, growth_rate, dropout_rate)
            n_feature += n_layers[1] * growth_rate
            self.trans2 = Transition(n_feature, dropout_rate)

            self.block3 = DenseBlock(
                n_layers[2], n_feature, bn_size, growth_rate, dropout_rate)
            n_feature += n_layers[2] * growth_rate
            self.trans3 = Transition(n_feature, dropout_rate)

            self.block4 = DenseBlock(
                n_layers[3], n_feature, bn_size, growth_rate, dropout_rate)
            n_feature += n_layers[3] * growth_rate
            self.bn4 = L.BatchNormalization(n_feature, dropout_rate)

            self.prob = L.Linear(None, n_class)

        self.n_class = n_class

    def __call__(self, x):
        """Feed-Forward calculation."""
        bs = len(x)
        h = F.relu(self.bn1(self.conv1(x)))

        h = self.block1(h)
        h = self.trans1(h)

        h = self.block2(h)
        h = self.trans2(h)

        h = self.block3(h)
        h = self.trans3(h)

        h = self.bn4(self.block4(h))
        h = F.relu(h)

        h = F.average_pooling_2d(h, h.shape[2])
        h = F.reshape(h, (bs, -1))

        return self.prob(h)


class DenseNetImagenet(chainer.Chain):

    """DenseNet for ImageNet.

    This class takes as input (n_channel, 224, 224) sized images.
    """

    def __init__(self, growth_rate=32, n_layers=(6, 12, 24, 16), init_features=64, bn_size=4, dropout_rate=0, n_class=1000):
        r"""Initialization of DenseNetImagenet.

        Args:
            growth_rate (int): growth_rate
            n_layers (int): # of layers for each block
            init_features (int): # of self.conv1's outputs' channels
            bn_size (int): bn_size
            dropout_rate (float): dropout rate \in [0, 1). If > 0,
                apply dropout
            n_class (int): # of class
        """
        super(DenseNetImagenet, self).__init__()
        with self.init_scope():
            initialW = initializers.HeNormal()
            self.conv1 = L.Convolution2D(
                None, init_features, 7, 2, 3, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(init_features)

            self.block1 = DenseBlock(
                n_layers[0], init_features, bn_size, growth_rate, dropout_rate)
            n_feature = init_features + n_layers[0] * growth_rate
            self.trans1 = Transition(n_feature, dropout_rate)

            self.block2 = DenseBlock(
                n_layers[1], n_feature, bn_size, growth_rate, dropout_rate)
            n_feature += n_layers[1] * growth_rate
            self.trans2 = Transition(n_feature, dropout_rate)

            self.block3 = DenseBlock(
                n_layers[2], n_feature, bn_size, growth_rate, dropout_rate)
            n_feature += n_layers[2] * growth_rate
            self.trans3 = Transition(n_feature, dropout_rate)

            self.block4 = DenseBlock(
                n_layers[3], n_feature, bn_size, growth_rate, dropout_rate)
            n_feature += n_layers[3] * growth_rate
            self.bn4 = L.BatchNormalization(n_feature)

            self.prob = L.Linear(None, n_class)

        self.n_class = n_class

    def __call__(self, x):
        """Feed-Forward calculation."""
        bs = len(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.block1(h)
        h = self.trans1(h)

        h = self.block2(h)
        h = self.trans2(h)

        h = self.block3(h)
        h = self.trans3(h)

        h = self.bn4(self.block4(h))
        h = F.relu(h)

        h = F.average_pooling_2d(h, 7)
        h = F.reshape(h, (bs, -1))

        return self.prob(h)


class DenseNet(chainer.Chain):

    """DenseBlock consists of any number of blocks.

    You specify the number of block via n_layers.
    """

    def __init__(self, growth_rate=32, n_layers=(6, 12, 24, 16), init_features=64, bn_size=4, dropout_rate=0, n_class=1000):
        """Initialization of DenseNet

        Args:
            growth_rate (int): default, 32
            n_layers (tuple of int): specify the number of blocks and
                the number of denselayer for each block
            init_features (int): the number of channel of output of conv1
            bn_size (int): default, 4
            dropout_rate (float, \in [0, 1)): if > 0, use dropout
            n_class (int): the number of target classes
        """
        super(DenseNet, self).__init__()

        self.n_block = len(n_layers)
        self.n_class = n_class

        with self.init_scope():
            initialW = initializers.HeNormal()
            self.conv1 = L.Convolution2D(
                None, init_features, 7, 2, 3, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(init_features)

            n_feature = init_features
            for block_idx, n_layer in enumerate(n_layers):
                setattr(self, 'block{}'.format(block_idx + 1), DenseBlock(n_layer,
                                                                          n_feature, bn_size, growth_rate, dropout_rate))
                n_feature += n_layer * growth_rate
                if block_idx + 1 < self.n_block:
                    setattr(self, 'trans{}'.format(
                        block_idx + 1), Transition(n_feature, dropout_rate=dropout_rate))
                else:
                    setattr(self, 'bn{}'.format(block_idx + 1),
                            L.BatchNormalization(n_feature))
            self.prob = L.Linear(None, self.n_class)

    def __call__(self, x):
        """Feed-Forward computation."""
        batch_size = len(x)
        h = F.relu(self.bn1(self.conv1(x)))

        for block_idx in range(1, self.n_block + 1):
            h = getattr(self, 'block{}'.format(block_idx))(h)
            if block_idx < self.n_block:
                h = getattr(self, 'trans{}'.format(block_idx))(h)
            else:
                h = getattr(self, 'bn{}'.format(block_idx))(h)

        h = F.reshape(h, (batch_size, -1))
        return self.prob(h)
