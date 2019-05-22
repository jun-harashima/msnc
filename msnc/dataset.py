# x: a token (e.g., character, subword, word).
# X: a sequence of tokens.
# X_set: a set of sequences.
# xs: a concatnation of sets.
# y: a number for an example
# ys: numbers for examples
import math
from msnc.util import Util


class Dataset():

    def __init__(self, examples, x_to_index=None, isregression=False):
        self.util = Util()
        self.pad_index = self.util.PAD_INDEX
        self.unk_index = self.util.UNK_INDEX

        X_sets = [[example['Xs'][i] for example in examples]
                  for i in range(len(examples[0]['Xs']))]

        self.x_to_index = x_to_index
        if x_to_index is None:
            self.x_to_index = []
            for i in range(len(examples[0]['Xs'])):
                xs = [x for X in X_sets[i] for x in X]
                self.x_to_index.append(self._make_index(xs))

        self.Xs = []
        self.raw_Xs = []  # for debug
        for i in range(len(examples[0]['Xs'])):
            self.Xs.append(self._degitize(X_sets[i], self.x_to_index[i]))
            self.raw_Xs.append(X_sets[i])

        if isregression:
            self.ys = [math.log10(example['y']) for example in examples]
        else:
            self.ys = [example['y'] for example in examples]

    def _make_index(self, xs):
        x_to_index = {'<PAD>': self.pad_index, '<UNK>': self.unk_index}
        for x in xs:
            if x not in x_to_index:
                x_to_index[x] = len(x_to_index)
        return x_to_index

    def _get_index(self, x, x_to_index):
        if x not in x_to_index:
            return x_to_index['<UNK>']
        return x_to_index[x]

    def _degitize(self, X_set, x_to_index):
        X = []
        for _X in X_set:
            _X = [self._get_index(x, x_to_index) for x in _X]
            X.append(_X)
        return X

    def split(self, batch_size):
        example_num = len(self.Xs[0])
        batch_num = int(example_num / batch_size)
        batches = [[] for _ in range(batch_num)]
        for X_set in self.Xs:
            self._append(batches, X_set, batch_size)
        self._append(batches, self.ys, batch_size)
        return batches

    def _append(self, batches, Z_set, batch_size):  # Z_set is X_set or ys
        for i in range(len(batches)):
            start = batch_size * i
            end = batch_size * (i + 1)
            batches[i].append(Z_set[start:end])
