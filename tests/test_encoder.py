import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
from msnc.encoder import Encoder


class TestEncoder(unittest.TestCase):

    def assertTorchEqual(self, x, y):
        self.assertTrue(torch.equal(x, y))

    def setUp(self):
        self.model = Encoder(4, 2, 2, 1)

        embedding_weight = torch.tensor([[0.0, 0.0],  # for <PAD>
                                         [1.0, 2.0],  # for <UNK>
                                         [3.0, 4.0],
                                         [5.0, 6.0]])
        self.embedding_weight = nn.Parameter(embedding_weight)

        self.X1 = ([2, 3, 2], [2, 1], [3, 1, 2, 3])
        self.X2 = [[3, 1, 2, 3], [2, 3, 2], [2, 1]]
        self.X3 = [[3, 1, 2, 3], [2, 3, 2, 0], [2, 1, 0, 0]]
        self.X4 = torch.tensor(self.X3, device=torch.device('cpu'))
        X5 = [[[5.0, 6.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
              [[3.0, 4.0], [5.0, 6.0], [3.0, 4.0], [0.0, 0.0]],
              [[3.0, 4.0], [1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]]
        self.X5 = torch.tensor(X5)
        self.X6 = nn.utils.rnn.PackedSequence(torch.tensor([[5.0, 6.0],
                                                            [3.0, 4.0],
                                                            [3.0, 4.0],
                                                            [1.0, 2.0],
                                                            [5.0, 6.0],
                                                            [1.0, 2.0],
                                                            [3.0, 4.0],
                                                            [3.0, 4.0],
                                                            [5.0, 6.0]]),
                                              torch.tensor([3, 3, 2, 1]))

        self.indices_before_sort = \
            torch.tensor([2, 0, 1], device=torch.device('cpu'))
        self.lengths_after_sort = [4, 3, 2]

    def test___init__(self):
        self.assertEqual(self.model.xdim, 4)
        self.assertEqual(self.model.edim, 2)
        self.assertEqual(self.model.hdim, 2)
        self.assertEqual(self.model.lnum, 1)
        self.assertEqual(self.model.pad_index, 0)
        self.assertTrue(self.model.use_lstm)
        self.assertTrue(self.model.use_bidirectional)
        self.assertFalse(self.model.use_cuda)
        self.assertIsInstance(self.model.embedding, nn.Module)
        self.assertIsInstance(self.model.rnn, nn.Module)

    def test__sort(self):
        X2, indices_before_sort, lengths_after_sort = self.model._sort(self.X1)
        self.assertEqual(X2, self.X2)
        self.assertTorchEqual(indices_before_sort, self.indices_before_sort)
        self.assertEqual(lengths_after_sort, self.lengths_after_sort)

    def test__pad(self):
        X3 = self.model._pad(self.X2, self.lengths_after_sort)
        self.assertEqual(X3, self.X3)

    def test__embed(self):
        with patch.object(self.model.embedding, 'weight',
                          self.embedding_weight):
            X5 = self.model._embed(self.X4)
            self.assertTorchEqual(X5, self.X5)

    def test__pack(self):
        X6 = self.model._pack(self.X5, self.lengths_after_sort)
        self.assertTorchEqual(X6.data, self.X6.data)
        self.assertTorchEqual(X6.batch_sizes, self.X6.batch_sizes)

    def test__rnn(self):
        H = self.model._rnn(self.X6)
        self.assertIsInstance(H, torch.Tensor)
        # (layer number, batch size, hidden dimension)
        self.assertEqual(H.shape, (2, 3, 2))

        model = Encoder(4, 2, 2, 1, use_lstm=True)
        H = model._rnn(self.X6)
        self.assertIsInstance(H, torch.Tensor)
        # (layer number, batch size, hidden dimension)
        self.assertEqual(H.shape, (2, 3, 2))

    def test__cat(self):
        H = self.model._rnn(self.X6)
        H2 = self.model._cat(H)
        self.assertTorchEqual(H2[0][:2], H[0][0])
        self.assertTorchEqual(H2[0][2:], H[1][0])

    def test__view(self):
        model = Encoder(4, 2, 2, 1, use_bidirectional=False)
        H = model._rnn(self.X6)
        H2 = model._view(H)
        self.assertTorchEqual(H2[0], H[0][0])
        self.assertTorchEqual(H2[1], H[0][1])
        self.assertTorchEqual(H2[2], H[0][2])

    def test__unsort(self):
        H = self.model._rnn(self.X6)
        H2 = self.model._cat(H)
        H3 = self.model._unsort(H2, self.indices_before_sort)
        self.assertTorchEqual(H3[0], H2[1])
        self.assertTorchEqual(H3[1], H2[2])
        self.assertTorchEqual(H3[2], H2[0])
