from typing import Dict
from typing import Any

import sys
import random
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from msnc.encoder import AverageEncoder
from msnc.encoder import RecurrentEncoder
from msnc.util import Util


class Model(nn.Module):

    def __init__(
        self,
        encoder_params,
        linear_params,
        epoch_num=100,
        checkpoint_interval=10,
        batch_size=32,
        seed=1,
    ):
        """Neural Network based classifier

        Arguments:
            encoder_params {Dict[str, Any]} -- encoder parameters
            linear_params {Dict[str, Any]} -- dense layer parameters

        Keyword Arguments:
            epoch_num {int} -- number of epochs (default: {100})
            checkpoint_interval {int} -- it creates checkpoints at {checkpoint_interval} (default: {10})  # NOQA
            batch_size {int} -- batch sizze (default: {32})
            seed {int} -- random seed (default: {1})
        """
        super(Model, self).__init__()
        random.seed(seed)
        torch.manual_seed(seed)
        self.util = Util()
        self.epoch_num = epoch_num
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size
        self.use_cuda = self.util.use_cuda
        self.encoders = self._init_encoders(encoder_params)
        self.linears = self._init_linears(linear_params)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        self.criterion = nn.NLLLoss()

        self._best_accuracy = None
        self._log = None

    def _init_encoders(self, encoder_params):
        encoders = []
        for params in encoder_params:
            if params.get('encoder') == 'average':
                encoders.append(AverageEncoder(**params))
            else:
                encoders.append(RecurrentEncoder(**params))

        return nn.ModuleList(encoders)

    def _init_linears(self, linear_params):
        linears = []
        for params in linear_params:
            linear = nn.Linear(params['indim'], params['outdim'])
            linear = linear.cuda() if self.use_cuda else linear
            linears.append(linear)
        return nn.ModuleList(linears)

    def run_training(self, output_dir, training_set, development_set=None):
        """run training procedure

        Arguments:
            output_dir {str} -- path to output dir
            TODO training_set {} --

        Keyword Arguments:
            TODO development_set {} --  (default: {None})
        """
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._output_path = output_path
        self._log = open(output_path / 'log.txt', 'w')
        self._best_accuracy = -float('inf')

        batches = training_set.split(self.batch_size)
        for epoch in range(1, self.epoch_num + 1):
            self.train()
            self._train(batches, epoch)
            if not self._ischeckpoint(epoch):
                continue
            self._save(epoch)
            if development_set is None:
                continue

            self.eval()
            self.run_evaluation(development_set)

        print('best_accuracy: {:3.2f}'.format(self._best_accuracy), file=sys.stderr)  # NOQA
        if self._log is not None:
            print('best_accuracy: {:3.2f}'.format(self._best_accuracy), file=self._log)  # NOQA
        self._log.close()

    def _train(self, batches, epoch):
        random.shuffle(batches)
        loss_sum = 0
        for *Xs, ys in batches:
            self.zero_grad()
            ys_hat = self(Xs)
            ys = self.util.tensorize(ys)
            loss = self.criterion(ys_hat, ys)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss
        print('epoch {:>3}\tloss {:6.2f}'.format(epoch, loss_sum), file=sys.stderr)  # NOQA
        if self._log is not None:
            print('epoch {:>3}\tloss {:6.2f}'.format(epoch, loss_sum), file=self._log)  # NOQA

    def _ischeckpoint(self, epoch):
        return epoch % self.checkpoint_interval == 0

    def _save(self, epoch):
        model_path = self._output_path / '{}.model'.format(epoch)
        torch.save(self.state_dict(), model_path.as_posix())

    def test(self, test_set):
        if len(test_set.Xs[0]) < self.batch_size:
            self.batch_size = len(test_set.Xs[0])
        batches = test_set.split(self.batch_size)
        results = self._test(batches)
        return results

    def _test(self, batches):
        results = []
        for *Xs, _ in batches:
            ys_hat = self(Xs)
            results.extend(ys_hat)
        return results

    def forward(self, Xs):
        Hs = []
        for i in range(len(self.encoders)):
            H = self.encoders[i](Xs[i])
            Hs.append(H)
        H = torch.cat(Hs, 1)
        for i in range(len(self.linears)):
            H = self.linears[i](H)
        return F.log_softmax(H, dim=1)

    def run_evaluation(self, test_set):
        ys_hat = [y_hat.argmax().item() for y_hat in self.test(test_set)]
        X_num = len(test_set.Xs)
        ok = 0
        for i in range(len(ys_hat)):
            for j in range(X_num):
                print("X{}:    ".format(j), test_set.Xs[j][i])
                print("raw_X{}:".format(j), test_set.raw_Xs[j][i])
            print("y:     ", test_set.ys[i])
            print("y_hat: ", ys_hat[i])
            if ys_hat[i] == test_set.ys[i]:
                ok += 1

        accuracy = ok / len(ys_hat)
        print('accuracy: {:3.2f}'.format(accuracy), file=sys.stderr)
        if self._log is not None:
            print('accuracy: {:3.2f}'.format(accuracy), file=self._log)

        if self._best_accuracy is not None:
            self._best_accuracy = max(self._best_accuracy, accuracy)
