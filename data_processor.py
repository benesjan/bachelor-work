import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
import re

input_data_path = '/home/honza/School/Bakalářka_Pr5/CNO_IPTC_train.txt'
# input_data_path = '/home/honza/School/Bakalářka_Pr5/test1'
output_matrix_path = '/home/honza/School/Bakalářka_Pr5/vectorizer'
output_topics_path = '/home/honza/School/Bakalářka_Pr5/topics'


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
    vectorizer = TfidfVectorizer()
    corpus, topics = build_corpus_and_topics(input_data_path)
    sparse_matrix = vectorizer.fit_transform(corpus)

    with open(output_matrix_path, 'wb') as handle:
        pickle.dump(sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_topics_path, 'wb') as handle:
        pickle.dump(topics, handle, protocol=pickle.HIGHEST_PROTOCOL)
