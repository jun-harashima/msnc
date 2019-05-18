import torch


class Util():

    PAD_INDEX = 0
    UNK_INDEX = 1

    def __init__(self):
        self.use_cuda = self._init_use_cuda()
        self.device = self._init_device()

    def _init_use_cuda(self):
        return torch.cuda.is_available()

    def _init_device(self):
        return torch.device('cuda' if self.use_cuda else 'cpu')

    def tensorize(self, Z):
        return torch.tensor(Z, device=self.device)
