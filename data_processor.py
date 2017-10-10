import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from data_utils import save_sparse_csr
import config


# Loads corpus into the memory. I could use yield to avoid this behavior but that would require processing the articles
# and article topics separately, since I don't want to write to global variables from within the function body
def build_corpus_and_topics(file_path):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'
    topics = {}
    corpus = []
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
                    topics_array = match_obj.group(2).split(' ')
                    article_id += 1
                    for topic in topics_array:
                        if topic in topics:
                            topics[topic].add(article_id)
                        else:
                            topics[topic] = {article_id}
                    if current_article:
                        print("Warning: Non empty article string")
                    print("Article with id " + str(article_id) + " loaded")
                    continue

            current_article += line

    return corpus, topics


if __name__ == '__main__':
    # vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, norm='l2', analyzer='word')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm='l2', analyzer='word')
    corpus, topics = build_corpus_and_topics(config.training_data_path)
    print("Articles loaded")
    print("Building the sparse matrix using the TfidfVectorizer")
    sparse_matrix = vectorizer.fit_transform(corpus) # Calculate

    print("Saving the matrix to file")
    save_sparse_csr(config.matrix_path, sparse_matrix)

    print("Saving the topics to file")
    with open(config.topics_path, 'wb') as handle:
        pickle.dump(topics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saving the vectorizer to file")
    with open(config.vectorizer_path, 'wb') as handle:
        pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
