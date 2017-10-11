import pickle
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

import config
from data_utils import save_sparse_csr


# Loads corpus into the memory. I could use yield to avoid this behavior but that would require processing the articles
# and article topics separately, since I don't want to write to global variables from within the function body
def build_corpus_and_topics(file_path):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'
    corpus = []
    topics = []
    current_article = ""
    article_id = -1  # I am not using the provided id because I need the id to match the index within corpus
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('<'):
                if line == '</article>\n':
                    corpus.append(current_article)
                    current_article = ""
                    continue

                match_obj = re.match(pattern, line)
                if match_obj:
                    topics.append(match_obj.group(2).split(' '))
                    article_id += 1
                    if current_article:
                        print("Warning: Non empty article string")
                    print("Article with id " + str(article_id) + " loaded")
                    continue

            current_article += line

    return corpus, topics


if __name__ == '__main__':
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, norm='l2', analyzer='word')
    binarizer = MultiLabelBinarizer()
    corpus, topics = build_corpus_and_topics(config.training_data_path)
    print("Articles loaded")

    if len(corpus) != len(topics):
        print("Error: matrix dimensions do not match")
        exit(1)

    print("Building the data matrix using the TfidfVectorizer")
    data_matrix = vectorizer.fit_transform(corpus)

    print("Building the label matrix by MultiLabelBinarizer")
    label_matrix = binarizer.fit_transform(topics)

    print("Saving the matrix to file")
    save_sparse_csr(config.data_matrix_path, data_matrix)

    print("Saving the topics to file")
    np.save(config.topics_matrix_path, label_matrix)

    print("Saving the vectorizer to file")
    with open(config.data_vectorizer_path, 'wb') as handle:
        pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saving the binarizer to file")
    with open(config.topic_binarizer_path, 'wb') as handle:
        pickle.dump(binarizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
