import re
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs

import config
from data_utils import load_pickle
from precision_measurement import r_cut


def build_topics_and_vectorized_paragraphs(raw_data_file_path, vectorizer, n_articles=1):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'
    articles, topics = [], []
    current_article = []

    articles_processed = 0
    with open(raw_data_file_path, 'r', encoding='utf-8') as handler:
        for line in handler:
            if line.startswith('<'):
                if line == '</article>\n':
                    articles.append(vectorizer.transform(current_article))
                    current_article = []
                    if articles_processed == n_articles:
                        break
                    continue

                match_obj = re.match(pattern, line)
                if match_obj:
                    topics.append(match_obj.group(2).split(' '))
                    articles_processed += 1
                    if current_article:
                        print("Warning: Non empty article string")
                    print(str(articles_processed) + ". article loaded")
                    continue

            current_article.append(line)

    if len(articles) != len(topics):
        raise ValueError("Error: matrix dimensions do not match")

    return articles, topics


if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    articles, topics = build_topics_and_vectorized_paragraphs(config.testing_data_path, vectorizer, 2000)

    for i in [-0.5, -0.25, 0, 0.25, 0.5]:
        print("Creating paragraph based predictions")
        article_topics = []
        for article in articles:
            y_dec = classifier.decision_function(article)
            y_rows = r_cut(y_dec, 1) + (y_dec > i)
            y_raw = np.sum(y_rows, 0)
            article_topics.append(y_raw)

        print("Merging predictions to single array")
        y_pred_raw = np.array(article_topics)
        y_pred = y_pred_raw > 0
        y_true = binarizer.transform(topics)

        P, R, F, S = prfs(y_true, y_pred, average="samples")
        print('i = %.2f, F1 = %.3f (P = %.3f, R = %.3f)' % (i, F, P, R))
