from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from custom_imports import config
from custom_imports.utils import load_pickle, build_corpus_and_topics, save_pickle

if __name__ == '__main__':
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    calibrated_classifier_one_class = CalibratedClassifierCV(LinearSVC(), cv=3)
    calibrated_classifier = OneVsRestClassifier(calibrated_classifier_one_class, n_jobs=4)

    corpus, topics = build_corpus_and_topics(config.training_data_path)

    print("Building the data matrix using the TfidfVectorizer")
    X = vectorizer.transform(corpus)

    print("Building the label matrix by MultiLabelBinarizer")
    Y = binarizer.transform(topics)

    print("Removing unnecessary variables")
    vectorizer, binarizer, corpus, topics = None, None, None, None

    print("Fitting the classifier")
    calibrated_classifier.fit(X, Y)

    print("Saving the classifier to file")
    save_pickle(config.calibrated_classifier_path, calibrated_classifier)  # Throws memory error
