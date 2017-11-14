from sklearn.calibration import CalibratedClassifierCV

from custom_imports import config
from custom_imports.utils import load_pickle, build_corpus_and_topics

if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    calibrated_classifier = CalibratedClassifierCV(classifier, cv='prefit')

    corpus, topics = build_corpus_and_topics(config.testing_data_path)

    print("Building the data matrix using the TfidfVectorizer")
    X = vectorizer.transform(corpus)

    print("Building the label matrix by MultiLabelBinarizer")
    Y = binarizer.transform(topics)

    print("Fitting the classifier")
    calibrated_classifier.fit(X, Y)
