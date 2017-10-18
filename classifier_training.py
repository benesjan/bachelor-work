# coding: utf-8
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import config
from data_utils import load_sparse_csr, save_pickle

if __name__ == '__main__':
    X = load_sparse_csr(config.data_matrix_path)
    Y = np.load(config.topics_matrix_path)

    # The process of data vectorization and learning could be streamlined by using Pipeline
    classifier = OneVsRestClassifier(LinearSVC(), n_jobs=4)
    print("Training the classifier")
    classifier.fit(X, Y)

    print("Saving the classifier to file")
    save_pickle(config.classifier_path, classifier)
