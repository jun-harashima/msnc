import torch
from msnc.dataset import Dataset
from msnc.model import Model


def construct(file_name):
    examples = []
    index = 0
    with open(file_name) as f:
        for line in f:
            y, *Xs = line.rstrip().split(',')
            Xs = [X.split() for X in Xs]
            examples.append({'index': index, 'Xs': Xs, 'y': y})
    return examples

training_examples = construct('scripts/training.csv')
development_examples = construct('scripts/training.csv')

training_dataset = Dataset(training_examples)
development_dataset = Dataset(development_examples, training_dataset.x_to_index)

xdims = [len(x_to_index) for x_to_index in training_dataset.x_to_index]

# xdim: Size of a vocabulay
# edim: Dimension of an embedding layer
# hdim: Dimension of a hidden layer
# lnum: Number of stacked RNN layers
encoder_params = [
    {'xdim': xdims[0], 'edim': 10, 'hdim': 4, 'lnum': 1, 'use_bidirectional': True},
    {'xdim': xdims[1], 'edim': 20, 'hdim': 2, 'lnum': 2, 'use_bidirectional': False},
]

first_indim = sum([param['hdim'] * (param['use_bidirectional'] + 1) for param in encoder_params])
first_outdim = 4
second_outdim = 2

# indim: Dimension of an input layer
# outdim: Dimension of an output layer
linear_params = [
    {'indim': first_indim, 'outdim': first_outdim},
    {'indim': first_outdim, 'outdim': second_outdim},
]

model = Model(encoder_params, linear_params)
model.run_training('scripts', training_dataset, development_dataset)
