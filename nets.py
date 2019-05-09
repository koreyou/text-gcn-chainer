import copy

import chainer
import chainer.functions as F
import numpy as np
import scipy.sparse as sp
from chainer import initializers
from chainer import reporter
from chainer.utils.sparse import CooMatrix

from graphs import to_chainer_sparse_variable


def sparse_to_gpu(x, device):
    x.data.data = chainer.backends.cuda.to_gpu(x.data.data, device=device)
    x.row = chainer.backends.cuda.to_gpu(x.row, device=device)
    x.col = chainer.backends.cuda.to_gpu(x.col, device=device)
    return x


def sparse_to_cpu(x):
    x.data.data = chainer.backends.cuda.to_cpu(x.data.data)
    x.row = chainer.backends.cuda.to_cpu(x.row)
    x.col = chainer.backends.cuda.to_cpu(x.col)
    return x


class TextGCN(chainer.Chain):
    def __init__(self, adj, labels, feat_size, dropout=0.5):
        super(TextGCN, self).__init__()
        n_class = np.max(labels) + 1
        initializer = initializers.HeUniform()
        with self.init_scope():
            self.gconv1 = GraphConvolution(adj.shape[1], feat_size)
            self.gconv2 = GraphConvolution(feat_size, n_class)
        # This Variable will not be updated because require_grad=False
        self.input = to_chainer_sparse_variable(
            sp.identity(adj.shape[1]))
        self.adj = adj
        self.labels = labels
        self.dropout = dropout

    def _forward(self):
        # deep copy object, shallow copy internal arrays
        x = copy.deepcopy(self.input)
        x.data = F.dropout(x.data, self.dropout)
        h = F.relu(self.gconv1(x, self.adj))
        h = F.dropout(h, self.dropout)
        out = self.gconv2(h, self.adj)
        return out

    def __call__(self, idx):
        out = self._forward()

        loss = F.softmax_cross_entropy(out[idx], self.labels[idx])
        accuracy = F.accuracy(out[idx], self.labels[idx])

        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': accuracy}, self)
        
        return loss

    def evaluate(self, idx):
        out = self._forward()

        loss = F.softmax_cross_entropy(out[idx], self.labels[idx])
        accuracy = F.accuracy(out[idx], self.labels[idx])

        return float(loss.data), float(accuracy.data)

    def predict(self, idx):
        out = self._forward()
        out = out[idx]
        pred = self.xp.argmax(out.data)
        return pred

    def predict_proba(self, idx):
        out = self._forward()
        out = out[idx]
        return out.data

    def to_gpu(self, device=None):
        self.adj = sparse_to_gpu(self.adj, device=device)
        self.input = sparse_to_gpu(self.input, device=device)
        self.labels = chainer.backends.cuda.to_gpu(self.labels, device=device)
        return super(TextGCN, self).to_gpu(device=device)

    def to_cpu(self):
        self.adj = sparse_to_cpu(self.adj)
        self.input = sparse_to_cpu(self.input)
        self.labels = chainer.backends.cuda.to_cpu(self.labels)
        return super(TextGCN, self).to_cpu()


class GraphConvolution(chainer.Link):
    def __init__(self, in_size, out_size=None, nobias=True, initialW=None,
                 initial_bias=None):
        super(GraphConvolution, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            if initialW is None:
                initialW = initializers.GlorotUniform()
            self.W = chainer.Parameter(initialW, (in_size, out_size))
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = chainer.Parameter(bias_initializer, out_size)

    def __call__(self, x, adj):
        if isinstance(x, chainer.utils.CooMatrix):
            x = F.sparse_matmul(x, self.W)
        else:
            x = F.matmul(x, self.W)
        output = F.sparse_matmul(adj, x)

        if self.b is not None:
            output += self.b

        return output
