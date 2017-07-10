# coding: utf-8
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers


class DenseLayer(chainer.Chain):
    """component of DenseBlock"""

    def __init__(self, in_ch, growth_rate, bn_size, dropout_rate=0.5):
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
        h = self.conv_1(F.relu(self.bn_1(x)))
        h = self.conv_2(F.relu(self.bn_2(h)))
        if self.dropout_rate > 0:
            h = F.dropout(h, self.dropout_rate)
        return F.concat((x, h))


class DenseBlock(chainer.Chain):
    """one block of DenseNet.

    DenseNet consists of 4 Blocks and 4 Transition
    """

    def __init__(self, n_layers, in_ch, bn_size, growth_rate, dropout_rate=0.5):
        super(DenseBlock, self).__init__()
        with self.init_scope():
            for i in range(n_layers):
                tmp_in_ch = in_ch + i * growth_rate
                setattr(self, 'denselayer{}'.format(
                    i + 1), DenseLayer(tmp_in_ch, growth_rate, bn_size, dropout_rate))
        self.n_layers = n_layers

    def __call__(self, x):
        h = x
        for i in range(1, self.n_layers + 1):
            h = getattr(self, 'denselayer{}'.format(i))(h)
        return h


class Transition(chainer.Chain):

    def __init__(self, in_ch, dropout_rate):
        super(Transition, self).__init__()
        with self.init_scope():
            initialW = initializers.HeNormal()
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(
                in_ch, in_ch, 1, initialW=initialW, nobias=True)
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        h = F.relu(self.bn(x))
        h = F.dropout(self.conv(h), self.dropout_rate)
        h = F.average_pooling_2d(h, 2)
        return h


class DenseNetCifar(chainer.Chain):

    def __init__(self, growth_rate=32, n_layers=(6, 12, 24, 16), init_features=64, bn_size=4, dropout_rate=0, n_class=1000):
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

    def __init__(self, growth_rate=32, n_layers=(6, 12, 24, 16), init_features=64, bn_size=4, dropout_rate=0, n_class=1000):
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


if __name__ == '__main__':
    import numpy as np
    # Feed Forward test for ImageNet
    x = np.random.randn(10, 3, 224, 224).astype(np.float32)
    x1 = np.random.randn(10, 16, 24, 24).astype(np.float32)
    dl = DenseLayer(3, 2, 2)
    out = dl(x)
    dl1 = DenseLayer(16, 2, 2)
    out1 = dl1(x1)
    print('out.shape: {}'.format(out.shape))
    print('out.shape: {}'.format(out1.shape))

    db = DenseBlock(2, 16, 4, 4, dropout_rate=0)
    x = np.random.randn(10, 16, 224, 224).astype(np.float32)
    print('db dict\n{}'.format(db._children))
    print('db.denselayer1.in_ch: {}'.format(db.denselayer1.in_ch))
    print('db.denselayer2.in_ch: {}'.format(db.denselayer2.in_ch))
    db_out = db(x)
    print('db_out.shape: {}'.format(db_out.shape))

    # m = DenseNetImagenet()
    # pred = m.forward(x)

    x = np.random.normal(size=(10, 3, 32, 32)).astype(np.float32)
    m = DenseNetCifar(n_class=100)
    pred = m.forward(x)
    print('pred.shape: {}'.format(pred.shape))