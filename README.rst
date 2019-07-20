====
msnc
====


.. image:: https://img.shields.io/pypi/v/msnc.svg
        :target: https://pypi.python.org/pypi/msnc

.. image:: https://img.shields.io/travis/jun-harashima/msnc.svg
        :target: https://travis-ci.org/jun-harashima/msnc

msnc is an acronym of multi-source neural classification.

Quick Start
===========

Installation
------------

Run this command in your terminal:

.. code-block:: bash

   $ pip install msnc

Pre-processing
--------------

See also ``scripts/sample.py``. First, prepare ``training_dataset`` and ``development_dataset`` as follows:

.. code-block:: python

    from msnc.dataset import Dataset

    training_dataset = Dataset(training_examples)
    development_dataset = Dataset(development_examples, training_dataset.x_to_index)

Note that ``{training, development}_examples`` are arrays which consist of ``{'index': index, 'Xs': Xs, 'y': y}``, where ``index`` (int), ``Xs`` (array of array of str), and ``y`` (int) represent an index, inputs, and output of an example, respectively.

Training
--------

Construct a ``Model`` object and train it as follows:

.. code-block:: python

    from msnc.model import Model

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
    model.run_training(output_dir, training_dataset, development_dataset)

The trained model is saved in the ``output_dir``. Note that you need the same number of parameters as the number of encoders (two in the above case). The following parameters for the encoders are optional.

- **use_bidirectional** - Use bidirectional RNN (default: ``True``)
- **use_lstm** - If ``True``, use LSTM, else GRU (Default: ``True``)

Test
----

TBD

Credits
=======

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
