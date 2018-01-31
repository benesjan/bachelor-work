import numpy as np

import config
from utils import load_pickle, save_pickle, save_sparse_csr, create_dir, build_topics_paragraphs_index_map


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

        paragraphs, topics, line_map = build_topics_paragraphs_index_map(data['text'])
        y_true = binarizer.transform(topics)

        print('Building the data matrix using the TfidfVectorizer')
        x = vectorizer.transform(paragraphs)

        print('Classifying...')
        y = classifier.predict_proba(x)

        print('Saving the data')
        save_sparse_csr(data['x'], x)
        np.save(data['y'], y)
        np.save(data['y_true'], y_true)
        save_pickle(data['line_map'], line_map)
