import chainer
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.datasets import fetch_20newsgroups


def moving_window_window_iterator(sentences, window):
    for sentence in sentences:
        for i in range(0, len(sentence) - window + 1):
            yield sentence[i:i + window]


def calc_pmi(X):
    Y = X.T.dot(X).astype(np.float32)
    Y_diag = Y.diagonal()
    Y.data /= Y_diag[Y.indices]
    Y.data *= X.shape[0]
    for col in range(Y.shape[1]):
        Y.data[Y.indptr[col]:Y.indptr[col + 1]] /= Y_diag[col]
    Y.data = np.maximum(0., np.log(Y.data))
    return Y


def create_text_adjacency_matrix(texts):
    """Create adjacency matrix from texts

    Arguments:
        texts (list of list of str): List of documents, each consisting
            of tokenized list of text

    Returns:
        adj (scipy.sparse.coo_matrix): (Node, Node) shape
            normalized adjency matrix.
    """
    transformer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_df=1.0, ngram_range=(1, 1), min_df=1, analyzer=lambda x: x)
    freq_doc = transformer.fit_transform(texts)

    freq_window = transformer.transform(
        moving_window_window_iterator(texts, 20))
    freq_window.data.fill(1)
    mat_pmi = calc_pmi(freq_window)

    adj = sp.bmat([[None, freq_doc], [freq_doc.T, mat_pmi]])

    adj.setdiag(np.ones([adj.shape[0]], dtype=adj.dtype))
    adj.eliminate_zeros()
    # it should already be COO, but behavior of bmat is not well documented
    # so apply it
    adj = adj.tocoo()

    adj = normalize(adj)

    return adj


def tokenize(text):
    return text.strip().split()


def load_20newsgroups(validation_ratio):
    """Load text network (20 news group)

    Returns:
        adj (chainer.utils.sparse.CooMatrix): (Node, Node) shape
            normalized adjency matrix.
        labels (np.ndarray): (Node, ) shape labels array
        idx_train (np.ndarray): Indices of the train
        idx_val (np.ndarray): Indices of val array
        idx_test (np.ndarray): Indices of test array
    """
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    adj = create_text_adjacency_matrix(
        [tokenize(t) for t in (train['data'] + test['data'])])

    n_train = int(len(train['data']) * (1.0 - validation_ratio))
    n_all = len(train['data']) + len(test['data'])
    idx_train = np.array(list(range(n_train)), np.int32)
    idx_val = np.array(list(range(n_train, len(train['data']))), np.int32)
    idx_test = np.array(list(range(len(train['data']), n_all)), np.int32)

    labels = np.concatenate(
        (train['target'], test['target'], np.full([adj.shape[0] - n_all], -1)))
    labels = labels.astype(np.int32)
    adj = to_chainer_sparse_variable(adj)

    return adj, labels, idx_train, idx_val, idx_test


def normalize(a):
    """ normalize adjacency matrix with normalization-trick.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    a += sp.eye(a.shape[0])
    rowsum = np.array(a.sum(1))
    rowsum_inv = np.power(rowsum, -1).flatten()
    rowsum_inv[np.isinf(rowsum_inv)] = 0.
    # ~D in the GCN paper
    d_tilde = sp.diags(rowsum_inv)
    # FIXME: multiply d_tilde from right as well?
    mx = d_tilde.dot(a)
    return mx


def to_chainer_sparse_variable(mat):
    mat = mat.tocoo().astype(np.float32)
    ind = np.argsort(mat.row)
    data = mat.data[ind]
    row = mat.row[ind]
    col = mat.col[ind]
    shape = mat.shape
    # check that adj's row indices are sorted
    assert np.all(np.diff(row) >= 0)
    return chainer.utils.CooMatrix(data, row, col, shape, order='C')
