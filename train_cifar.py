from __future__ import print_function
import argparse

import matplotlib
matplotlib.use('Agg')

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer.datasets import split_dataset_random
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

from chainercv.datasets import TransformDataset

from densenet import DenseNetCifar


def transform(in_data):
    import random
    import numpy as np
    img, label = in_data
    pad_x = np.zeros((3, 40, 40), dtype=np.float32)
    pad_x[:, 4:36, 4:36] = np.asarray(img)
    top = random.randint(0, 8)
    left = random.randint(0, 8)
    x = pad_x[:, top:top + 32, left:left + 32]
    if random.randint(0, 1):
        x = x[:, :, ::-1]
    return x, label


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', '-s', type=int, default=123,
                        help='seed for split dataset into train & validation')
    parser.add_argument('--augment', '-a', type=bool, default=True,
                        help='whether augment dataset or not')
    parser.add_argument('--parallel', '-p', type=bool, default=True,
                        help='use multiprocess iterator or not')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')
    if args.augment:
        train = TransformDataset(train, transform)
    else:
        train, val = split_dataset_random(train, 45000, seed=args.seed)

    model = L.Classifier(DenseNetCifar(n_class=class_labels))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.NesterovAG(lr=args.learnrate, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    if args.parallel:
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=2)
        test_iter = chainer.iterators.MultiprocessIterator(
            test, args.batchsize, repeat=False, shuffle=False, n_processes=2)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.snapshot_object(model.predictor,
                                              filename='densenet.model'), trigger=(args.epoch, 'epoch'))

    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'lr', 'elapsed_time']))

    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss', 'validation/main/loss'], file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        y_keys=['main/accuracy', 'validation/main/accuracy'], file_name='accuracy.png'))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
