import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

import config
from data_utils import save_sparse_csr, save_pickle


# Returns data and topics lists in a not-vectorized form
def build_corpus_and_topics(raw_data_file_path, n_articles=-1):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'
    corpus, topics = [], []
    current_article = ""

    articles_processed = 0
    with open(raw_data_file_path, 'r', encoding='utf-8') as handler:
        for line in handler:
            if line.startswith('<'):
                if line == '</article>\n':
                    corpus.append(current_article)
                    if articles_processed == n_articles:
                        break
                    current_article = ""
                    continue

                match_obj = re.match(pattern, line)
                if match_obj:
                    topics.append(match_obj.group(2).split(' '))
                    articles_processed += 1
                    if current_article:
                        print("Warning: Non empty article string")
                    print(str(articles_processed) + ". article loaded")
                    continue

            current_article += line

    if len(corpus) != len(topics):
        raise ValueError("Error: matrix dimensions do not match")

    return corpus, topics


# Creates 2 files. One contains held-out data and the other one training data
def separate_data(source_file_path, held_out_path, training_data_path, nth_article=10):
    append_training = True
    articles_processed = 0

    with open(source_file_path, 'r', encoding='utf-8') as source, \
            open(held_out_path, 'a', encoding='utf-8') as held_out, \
            open(training_data_path, 'a', encoding='utf-8') as training:
        for line in source:
            if line.startswith('<article') and articles_processed % nth_article == 0:
                append_training = False

            if append_training:
                training.write(line)
            else:
                held_out.write(line)

            if line == '</article>\n':
                articles_processed += 1
                append_training = True


if __name__ == '__main__':
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, norm='l2', analyzer='word')
    binarizer = MultiLabelBinarizer()

    print("Separating held-out from training data")
    separate_data(config.training_data_raw_path, config.held_out_data_path, config.training_data_path)

    corpus, topics = build_corpus_and_topics(config.training_data_path)
    print("Articles loaded")

    print("Building the data matrix using the TfidfVectorizer")
    data_matrix = vectorizer.fit_transform(corpus)

    print("Building the label matrix by MultiLabelBinarizer")
    label_matrix = binarizer.fit_transform(topics)

    print("Saving the matrix to file")
    save_sparse_csr(config.data_matrix_path, data_matrix)

    print("Saving the topics to file")
    np.save(config.topics_matrix_path, label_matrix)

    print("Saving the vectorizer to file")
    save_pickle(config.data_vectorizer_path, vectorizer)

    print("Saving the binarizer to file")
    save_pickle(config.topic_binarizer_path, binarizer)
