from re import match

import numpy as np

from custom_imports import config
from custom_imports.utils import load_pickle, save_pickle, save_sparse_csr


def build_topics_paragraphs_index_map(raw_data_file_path, n_articles=-1):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'

    articles, topics, line_map = [], [], []
    articles_processed, paragraph_index = 0, 0

    with open(raw_data_file_path, 'r', encoding='utf-8') as handler:
        for line in handler:
            if line.startswith('<'):
                if line == '</article>\n':
                    line_map.append(paragraph_index)
                    # number of elements in line_map should be even since it contains start-end pairs
                    assert len(line_map) % 2 == 0
                    # End of article
                    if articles_processed == n_articles:
                        break
                    continue

                match_obj = match(pattern, line)
                if match_obj:
                    line_map.append(paragraph_index)
                    assert len(line_map) % 2 == 1
                    # First line of article
                    topics.append(match_obj.group(2).split(' '))
                    articles_processed += 1
                    print(str(articles_processed) + ". article loaded")
                    continue

            paragraph_index += 1
            articles.append(line)

    if len(line_map) != (len(topics) * 2):
        raise ValueError("Error: matrix dimensions do not match")

    return articles, topics, line_map


if __name__ == '__main__':
    classifier = load_pickle(config.classifier)
    vectorizer = load_pickle(config.vectorizer)
    binarizer = load_pickle(config.binarizer)

    articles, topics, line_map = build_topics_paragraphs_index_map(config.train_data)
    y_true = binarizer.transform(topics)

    print("Building the data matrix using the TfidfVectorizer")
    x = vectorizer.transform(articles)

    print("Classifying...")
    y_pred = classifier.predict_proba(x)

    print("Saving the data")
    np.save(config.y_par, y_pred)
    np.save(config.y_par_true, y_true)
    save_sparse_csr(config.x_par, x)
    save_pickle(config.line_map, line_map)
