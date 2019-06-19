import unittest
from msnc.model import Model
from msnc.encoder import AverageEncoder
from msnc.encoder import RecurrentEncoder


class TestModel(unittest.TestCase):

    def setUp(self):
        self.encoder_params_rnn = [
            {
                'xdim': 200, 'edim': 100, 'hdim': 50,
                'lnum': 2, 'use_lstm': True,
                'use_bidirectional': True
            },
        ]

        self.linear_params_rnn = [
            {
                'indim': 2*50, 'outdim': 40
            },
            {
                'indim': 40, 'outdim': 20
            },
        ]

        self.encoder_params_avg = [
            {
                'xdim': 200, 'edim': 100,
                'encoder': 'average'
            },
        ]

        self.linear_params_avg = [
            {
                'indim': 100, 'outdim': 40
            },
            {
                'indim': 40, 'outdim': 20
            },
        ]

    def test___init__with_rnn(self):
        model = Model(self.encoder_params_rnn, self.linear_params_rnn)
        self.assertIsInstance(model.encoders[0], RecurrentEncoder)

    def test___init__with_avg(self):
        model = Model(self.encoder_params_avg, self.linear_params_avg)
        self.assertIsInstance(model.encoders[0], AverageEncoder)

    def test__set_log_line(self):
        model = Model(self.encoder_params_rnn, self.linear_params_rnn)

        log_line = model._set_log_line(0.100, 10)
        self.assertEqual(log_line, ' dev_accuracy: 0.10   epoch: 10')

        log_line = model._set_log_line(0.100, 10, 'note')
        self.assertEqual(log_line, 'note dev_accuracy: 0.10   epoch: 10')
