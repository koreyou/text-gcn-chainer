import chainer
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.datasets import fetch_20newsgroups

from nlp_utils import tokenize


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
    # The authors removed words occuring less than 5 times. It is not directory
    # applicable to min_df, so I set bit smaller value
    transformer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_df=1.0, ngram_range=(1, 1), min_df=3, analyzer='word',
        preprocessor=lambda x: x, tokenizer=lambda x: x,
        norm=None, smooth_idf=False
    )
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

    return adj


def load_20newsgroups(validation_ratio, normalization):
    """Load text network (20 news group)

    Arguments:
        validation_ratio (float): Ratio of validation split
        normalization (str): Variant of normalization method to use.

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
    if normalization == 'gcn':
        adj = normalize(adj)
    else:
        adj = normalize_pygcn(adj)
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


def normalize_pygcn(a):
    """ normalize adjacency matrix with normalization-trick. This variant
    is proposed in https://github.com/tkipf/pygcn .
    Refer https://github.com/tkipf/pygcn/issues/11 for the author's comment.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    # no need to add identity matrix because self connection has already been added
    # a += sp.eye(a.shape[0])
    rowsum = np.array(a.sum(1))
    rowsum_inv = np.power(rowsum, -1).flatten()
    rowsum_inv[np.isinf(rowsum_inv)] = 0.
    # ~D in the GCN paper
    d_tilde = sp.diags(rowsum_inv)
    return d_tilde.dot(a)


def normalize(adj):
    """ normalize adjacency matrix with normalization-trick that is faithful to
    the original paper.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    # no need to add identity matrix because self connection has already been added
    # a += sp.eye(a.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # ~D in the GCN paper
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


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
