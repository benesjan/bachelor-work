import config

from data_utils import load_pickle
from data_processor import build_corpus_and_topics
from sklearn.metrics import precision_recall_fscore_support

if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    corpus, topics = build_corpus_and_topics(config.testing_data_path)

    X = vectorizer.transform(corpus)
    Y_true = binarizer.transform(topics)

    Y_pred = classifier.predict(X)

