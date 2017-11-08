import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs

from custom_imports import config
from custom_imports.utils import load_pickle, build_topics_and_paragraphs


def compute_weights(article):
    weights = []
    for paragraph in article['paragraphs']:
        weights.append(len(paragraph) / article['length'])
        # weights.append(1)
    return np.array(weights)


if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    articles, topics = build_topics_and_paragraphs(config.testing_data_path, 2000)
    print("Creating paragraph based predictions")

    article_topics = []
    for article in articles:
        y_dec_article = classifier.decision_function(vectorizer.transform(article['paragraphs']))
        weights = compute_weights(article)

        # vectorized way of multiplying the matrix rows by weights and summing the rows into one
        y_dec_weighted = np.dot(np.transpose(y_dec_article), np.transpose(weights))

        article_topics.append(y_dec_weighted)

    print("Merging predictions to single array")
    y_pred = np.array(article_topics) > -0.67  # Ideal threshold
    y_true = binarizer.transform(topics)

    P, R, F, S = prfs(y_true, y_pred, average="samples")
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
