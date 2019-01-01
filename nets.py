import chainer
import chainer.functions as F
import numpy as np
from chainer import initializers
from chainer import reporter


class TextGCN(chainer.Chain):
    def __init__(self, adj, labels, feat_size, dropout=0.5):
        super(TextGCN, self).__init__()
        n_class = np.max(labels) + 1
        initializer = initializers.HeUniform()
        with self.init_scope():
            self.gconv1 = GraphConvolution(feat_size, feat_size, noweight=True)
            self.gconv2 = GraphConvolution(feat_size, n_class)
            self.input_repr = chainer.Parameter(
                initializer, (adj.shape[0], feat_size))
        self.adj = adj
        self.labels = labels
        self.dropout = dropout

    def _forward(self):
        h = F.relu(self.gconv1(self.input_repr, self.adj))
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
        adj = self.adj
        adj.data.data = chainer.backends.cuda.to_gpu(adj.data.data, device=device)
        adj.row = chainer.backends.cuda.to_gpu(adj.row, device=device)
        adj.col = chainer.backends.cuda.to_gpu(adj.col, device=device)
        self.adj = adj
        self.labels = chainer.backends.cuda.to_gpu(self.labels, device=device)
        return super(TextGCN, self).to_gpu(device=device)

    def to_cpu(self):
        adj = self.adj
        adj.data.data = chainer.backends.cuda.to_cpu(adj.data.data)
        adj.row = chainer.backends.cuda.to_cpu(adj.row)
        adj.col = chainer.backends.cuda.to_cpu(adj.col)
        self.labels = chainer.backends.cuda.to_cpu(self.labels)
        return super(TextGCN, self).to_cpu()


class GraphConvolution(chainer.Link):
    def __init__(self, in_size, out_size=None, noweight=False,
                 nobias=False, initialW=None, initial_bias=None):
        super(GraphConvolution, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            if noweight:
                self.W = None
            else:
                if initialW is None:
                    initialW = initializers.HeUniform()
                self.W = chainer.Parameter(initialW, (in_size, out_size))
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = chainer.Parameter(bias_initializer, out_size)

    def __call__(self, x, adj):
        if self.W is not None:
            x = F.matmul(x, self.W)
        output = F.sparse_matmul(adj, x)

        if self.b is not None:
            output += self.b

        return output
