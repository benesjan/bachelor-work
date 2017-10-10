import pickle
import config
from sys import exit
from  data_utils import load_sparse_csr

if __name__ == '__main__':
    try:
        matrix = load_sparse_csr(config.matrix_path)

        with open(config.topics_path, 'rb') as f:
            topics = pickle.load(f)

    except FileNotFoundError as e:
        print("File loading failed: " + e.filename)
        exit(1)
