====
msnc
====


.. image:: https://img.shields.io/pypi/v/msnc.svg
        :target: https://pypi.python.org/pypi/msnc

.. image:: https://img.shields.io/travis/jun-harashima/msnc.svg
        :target: https://travis-ci.org/jun-harashima/msnc

msnc is an acronym of multi-source neural classification.

Quick Start
-----------

Installation
^^^^^^^^^^^^

Run this command in your terminal:

.. code-block:: bash

   $ pip install msnc

Pre-processing
--------------

TBD

Training
--------

Construct a ``Model`` object and train it as follows:

.. code-block:: python

    xdims = [len(x_to_index) for x_to_index in training_dataset.x_to_index]

    # xdim: Size of a vocabulay
    # edim: Dimension of an embedding layer
    # hdim: Dimension of a hidden layer
    # lnum: Number of stacked RNN layers
    encoder_params = [
        {'xdim': xdims[0], 'edim': 100, 'hdim': 20, 'lnum': 2},
        {'xdim': xdims[1], 'edim': 100, 'hdim': 20, 'lnum': 2},
    ]

    # indim: Dimension of an input layer
    # outdim: Dimension of an output layer
    linear_params = [
        {'indim': 40, 'outdim': 20},
        {'indim': 20, 'outdim': 20},
    ]

    model = Model(encoder_params, linear_params)
    model.run_training(output_dir, training_dataset, development_dataset)

The trained model is saved in the ``output_dir``.

For the Encoders, you can also use the following parameters:

- **use_bidirectional** - Use bidirectional RNN (default: ``True``)
- **use_lstm** - If ``True``, use LSTM, else GRU (Default: ``True``)

Pre-processing
--------------

Test

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
