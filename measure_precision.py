# coding: utf-8
from sklearn.metrics import precision_recall_fscore_support as prfs

from custom_imports import config
from custom_imports.utils import load_pickle, build_corpus_and_topics, r_cut, yes_or_no


def predict_tuned(x, classifier):
    print('Classifying the data')
    y_dec = classifier.decision_function(x)

    # Ensures at least 1 predicted topic for each article
    y_pred_min_topics = r_cut(y_dec, 1)

    # Returns matrix where each elements is set to True if the element's value is bigger than threshold
    y_pred_threshold = y_dec > -0.32  # -0.32 for 1 min topic (tested on held-out data)

    return y_pred_min_topics + y_pred_threshold


def predict_tuned_calibrated(x, classifier):
    print('Classifying the data')
    y_prob = classifier.predict_proba(x)

    # Ensures at least 1 predicted topic for each article
    y_pred_min_topics = r_cut(y_prob, 1)

    # Returns matrix where each elements is set to True if the element's value is bigger than threshold
    y_pred_threshold = y_prob > 0.3

    return y_pred_min_topics + y_pred_threshold


if __name__ == '__main__':
    calibrated = yes_or_no("Do you want to use calibrated classifier?")

    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    corpus, topics = build_corpus_and_topics(config.testing_data_path)

    print('Transforming corpus by vectorizer')
    x = vectorizer.transform(corpus)
    print('Transforming article topics by binarizer')
    y_true = binarizer.transform(topics)

    if calibrated:
        print('Loading calibrated classifier')
        classifier = load_pickle(config.calibrated_classifier_path)
        y_pred = predict_tuned_calibrated(x, classifier)
    else:
        print('Loading uncalibrated classifier')
        classifier = load_pickle(config.classifier_path)
        y_pred = predict_tuned(x, classifier)

    P, R, F, S = prfs(y_true, y_pred, average='samples')
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))