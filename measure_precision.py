# coding: utf-8
from sklearn.metrics import precision_recall_fscore_support as prfs

from create_classifier_paragraphs import process_y, threshold_half_max
from custom_imports import config
from custom_imports.utils import load_pickle, build_corpus_and_topics, r_cut, choose_option, load_sparse_csr


def predict_tuned(x, classifier, threshold):
    print('Classifying the data')
    y_prob = classifier.predict_proba(x)

    # Ensures at least 1 predicted topic for each article
    y_pred_min_topics = r_cut(y_prob, 1)

    # Returns matrix where each elements is set to True if the element's value is bigger than threshold
    y_pred_threshold = y_prob > threshold

    return y_pred_min_topics + y_pred_threshold


if __name__ == '__main__':
    if choose_option('Do you want to use paragraphs trained classifier or the  article trained version?', 'p', 'a'):
        threshold = 0.39

        data = config.get_par_data('test')

        print('Loading the paragraph trained classifier')
        classifier = load_pickle(config.classifier_par)

        y_true = process_y(data, threshold_half_max)

        print("Loading x")
        x = load_sparse_csr(data['x'])
    else:
        threshold = 0.3

        vectorizer = load_pickle(config.vectorizer)
        binarizer = load_pickle(config.binarizer)

        print('Loading the classifier')
        classifier = load_pickle(config.classifier)

        corpus, topics = build_corpus_and_topics(config.data['test'])

        print('Transforming corpus by vectorizer')
        x = vectorizer.transform(corpus)
        print('Transforming article topics by binarizer')
        y_true = binarizer.transform(topics)

    y_pred = predict_tuned(x, classifier, threshold)

    P, R, F, S = prfs(y_true, y_pred, average='samples')
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
