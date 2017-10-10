from numpy import savez, load
from scipy.sparse import csr_matrix


def save_sparse_csr(filename, array):
    savez(filename, data=array.data, indices=array.indices,
          indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])