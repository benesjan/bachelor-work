import numpy as np
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import config
from data_utils import load_sparse_csr

if __name__ == '__main__':
    X = load_sparse_csr(config.data_matrix_path)
    Y = np.load(config.topics_matrix_path)

    # The process of vectorizing and learning could be streamlined by using Pipeline
    clf = OneVsRestClassifier(LinearSVC())
    print("Training the classifier")
    clf.fit(X, Y)

    print("Saving the classifier to file")
    with open(config.classifier_path, 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
