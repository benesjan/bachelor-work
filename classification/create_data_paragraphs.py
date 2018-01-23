from re import match
import numpy as np

import config
from utils import load_pickle, save_pickle, save_sparse_csr, create_dir


def build_topics_paragraphs_index_map(data_path, n_articles=-1):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'

    articles, topics, line_map = [], [], []
    articles_processed, paragraph_index = 0, 0

    with open(data_path, 'r', encoding='utf-8') as handler:
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
                    print(str(articles_processed) + '. article loaded')
                    continue

            paragraph_index += 1
            articles.append(line)

    if len(line_map) != (len(topics) * 2):
        raise ValueError('Error: matrix dimensions do not match')

    return articles, topics, line_map


def get_next_data(data_names):
    for name in data_names:
        yield config.get_par_data(name)


if __name__ == '__main__':
    print('Loading the instances')
    classifier = load_pickle(config.classifier)
    vectorizer = load_pickle(config.vectorizer)
    binarizer = load_pickle(config.binarizer)

    for data in get_next_data(config.data.keys()):
        print('Processing ' + data['name'] + ' data')

        create_dir(data['dir'])

        articles, topics, line_map = build_topics_paragraphs_index_map(data['text'])
        y_true = binarizer.transform(topics)

        print('Building the data matrix using the TfidfVectorizer')
        x = vectorizer.transform(articles)

        print('Classifying...')
        y = classifier.predict_proba(x)

        print('Saving the data')
        save_sparse_csr(data['x'], x)
        np.save(data['y'], y)
        np.save(data['y_true'], y_true)
        save_pickle(data['line_map'], line_map)
