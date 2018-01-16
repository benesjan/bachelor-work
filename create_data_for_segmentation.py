from create_paragraph_data import build_topics_paragraphs_index_map
from custom_imports.utils import load_pickle, create_dir, save_sparse_csr, save_pickle
from custom_imports import config


def get_next_data(data_names):
    for name in data_names:
        yield config.get_seg_data(name)


if __name__ == '__main__':
    print('Loading the instances')
    vectorizer = load_pickle(config.vectorizer)

    for data in get_next_data(['held_out', 'test']):
        print('Processing ' + data['name'] + ' data')

        create_dir(data['dir'])

        articles, ignored, line_map = build_topics_paragraphs_index_map(data['text'])

        print('Building the data matrix using the TfidfVectorizer')
        x = vectorizer.transform(articles)

        print('Saving the data')
        save_sparse_csr(data['x'], x)
        save_pickle(data['line_map'], line_map)
