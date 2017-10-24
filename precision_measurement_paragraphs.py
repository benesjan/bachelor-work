import re

import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs

import config
from data_utils import load_pickle


def build_topics_and_paragraphs(raw_data_file_path, n_articles=1):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'
    articles, topics = [], []
    current_article = []

    articles_processed = 0
    article_length = 0
    with open(raw_data_file_path, 'r', encoding='utf-8') as handler:
        for line in handler:
            if line.startswith('<'):
                if line == '</article>\n':
                    articles.append({
                        'paragraphs': current_article,
                        'length': article_length
                    })
                    if articles_processed == n_articles:
                        break
                    article_length = 0
                    current_article = []
                    continue

                match_obj = re.match(pattern, line)
                if match_obj:
                    topics.append(match_obj.group(2).split(' '))
                    articles_processed += 1
                    if current_article:
                        print("Warning: Non empty article string")
                    print(str(articles_processed) + ". article loaded")
                    continue

            article_length += len(line)
            current_article.append(line)

    if len(articles) != len(topics):
        raise ValueError("Error: matrix dimensions do not match")

    return articles, topics


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
        y_dec_weighted = np.dot(np.transpose(y_dec_article > 0.04), np.transpose(weights))

        article_topics.append(y_dec_weighted)
        # article_topics.append(np.sum(y_dec_article > threshold, 0))

    print("Merging predictions to single array")
    y_pred = np.array(article_topics) > 0
    y_true = binarizer.transform(topics)

    P, R, F, S = prfs(y_true, y_pred, average="samples")
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
