import pickle

from numpy import savez, load
from scipy.sparse import csr_matrix


def save_sparse_csr(filename, array):
    savez(filename, data=array.data, indices=array.indices,
          indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def save_pickle(file_path, object_to_save):
    with open(file_path, 'wb') as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
