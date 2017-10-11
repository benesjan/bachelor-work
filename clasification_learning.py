import pickle
from sys import exit

import config
from data_utils import load_sparse_csr

if __name__ == '__main__':
    try:
        matrix = load_sparse_csr(config.data_matrix_path)

        with open(config.topics_matrix_path, 'rb') as f:
            topics = pickle.load(f)

    except FileNotFoundError as e:
        print("File loading failed: " + e.filename)
        exit(1)
