import random
import pathlib
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from msnc.encoder import AverageEncoder
from msnc.encoder import RecurrentEncoder
from msnc.util import Util


logger = logging.getLogger(__file__)


class Model(nn.Module):

    def __init__(
        self,
        encoder_params,
        linear_params,
        epoch_num=100,
        checkpoint_interval=10,
        batch_size=32,
        seed=1,
        save_best_model=True,
    ):
        """Neural network based classifier

        Arguments:
            encoder_params {Dict[str, Any]} -- encoder parameters
            linear_params {Dict[str, Any]} -- dense layer parameters

        Keyword arguments:
            epoch_num {int} -- number of epochs (default: {100})
            checkpoint_interval {int} -- it creates checkpoints at {checkpoint_interval} (default: {10})  # NOQA
            batch_size {int} -- batch size (default: {32})
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

        self._best_dev_accuracy = None
        self._best_epoch = None
        self._log = None
        self._save_best_model = save_best_model

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

    def run_training(
        self,
        output_dir,
        training_set,
        development_set=None,
    ):
        """Run training procedure

        Arguments:
            output_dir_path {str} -- path to output dir
            TODO training_set {} -- dataset for training

        Keyword Arguments:
            TODO development_set {} --  dataset for validating (default: {None})  # NOQA
        """
        self._output_dir_path = pathlib.Path(output_dir)
        self._best_dev_accuracy = -float('inf')
        self._best_epoch = 0

        batches = training_set.split(self.batch_size)
        for epoch in range(1, self.epoch_num + 1):
            self.train()
            self._train(batches, epoch)

            if not self._ischeckpoint(epoch):
                continue
            model_file_name = '{:04d}.model'.format(epoch)
            self._save(model_file_name)

            if development_set is not None:
                dev_accuracy = self.run_evaluation(development_set)
                log_line = 'dev accuracy: {:3.2f}'.format(dev_accuracy)
                log_line += '   epoch: {}'.format(epoch)
                logger.info(log_line)

                if not self.is_best(dev_accuracy):
                    continue

                # When the new model outperforms the previous ones
                log_line = '[new best] dev accuracy: {:3.2f}'.format(dev_accuracy)  # NOQA
                log_line += '   epoch: {}'.format(epoch)

                # Save epoch and performance information
                self._best_epoch = epoch
                self._best_dev_accuracy = dev_accuracy

                if self._save_best_model:
                    self._save('best.model')
                    logger.debug('Update best model')

                logging.info(log_line)

        log_line = '[best] dev_accuracy: {:3.2f}'.format(self._best_dev_accuracy)  # NOQA
        log_line += '   epoch: {}'.format(self._best_epoch)
        logger.info(log_line)

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

        logger.info('epoch {:>3}\tloss {:6.2f}'.format(epoch, loss_sum))

    def _ischeckpoint(self, epoch):
        return epoch % self.checkpoint_interval == 0

    def _save(self, model_file_name: str):
        model_path = self._output_dir_path / model_file_name
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
        self.eval()

        ys_hat = [y_hat.argmax().item() for y_hat in self.test(test_set)]
        X_num = len(test_set.Xs)
        ok = .0
        for i in range(len(ys_hat)):
            for j in range(X_num):
                logger.debug("X{}:    ".format(j) + str(test_set.Xs[j][i]))
                logger.debug("raw_X{}:".format(j) + str(test_set.raw_Xs[j][i]))
            logger.debug("y:     " + str(test_set.ys[i]))
            logger.debug("y_hat: " + str(ys_hat[i]))
            if ys_hat[i] == test_set.ys[i]:
                ok += 1

        accuracy = ok / len(ys_hat)
        return accuracy

    def is_best(self, dev_accuracy):
        if self._best_dev_accuracy is None:
            return False
        return dev_accuracy > self._best_dev_accuracy
