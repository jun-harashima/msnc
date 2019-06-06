import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
from msnc.encoder import AverageEncoder


class TestAverageEncoder(unittest.TestCase):

    def assertTorchEqual(self, x, y):
        self.assertTrue(torch.equal(x, y))

    def setUp(self):
        self.model = AverageEncoder(4, 2, 1)

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

    def test___init__(self):
        self.assertEqual(self.model.xdim, 4)
        self.assertEqual(self.model.edim, 2)
        self.assertEqual(self.model.pad_index, 0)
        self.assertFalse(self.model.use_cuda)
        self.assertIsInstance(self.model.embedding, nn.Module)

    def test__embed(self):
        with patch.object(self.model.embedding, 'weight',
                          self.embedding_weight):
            X5 = self.model._embed(self.X4)
            self.assertTorchEqual(X5, self.X5)

    def test_average(self):
        print(self.model(self.X1))

        ground_truth = []
        for data in self.X1:
            emb = self.model.embedding(torch.tensor([data]))
            ground_truth.append(emb.mean(dim=1))
        ground_truth = torch.cat(ground_truth)

        self.assertTorchEqual(ground_truth, self.model(self.X1))
