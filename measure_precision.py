# coding: utf-8
from sklearn.metrics import precision_recall_fscore_support as prfs

from custom_imports import config
from custom_imports.classifier_functions import predict_tuned
from custom_imports.utils import load_pickle, build_corpus_and_topics

if __name__ == '__main__':
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    corpus, topics = build_corpus_and_topics(config.testing_data_path)

    print("Transforming corpus by vectorizer")
    x = vectorizer.transform(corpus)
    print("Transforming article topics by binarizer")
    y_true = binarizer.transform(topics)

    y_pred = predict_tuned(x)

    P, R, F, S = prfs(y_true, y_pred, average="samples")
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
