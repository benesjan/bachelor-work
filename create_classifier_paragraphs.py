import scipy as sp
from re import match
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from custom_imports import config
from custom_imports.utils import load_pickle, r_cut, save_pickle


def build_topics_paragraphs_index_map(raw_data_file_path, n_articles=-1):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'

    articles_paragraphs, topics, article_map = [], [], []
    articles_processed, paragraph_index = 0, 0

    with open(raw_data_file_path, 'r', encoding='utf-8') as handler:
        for line in handler:
            if line.startswith('<'):
                if line == '</article>\n':
                    article_map.append(paragraph_index - 1)
                    # number of elements in article_map should be even since it contains start-end pairs
                    assert len(article_map) % 2 == 0
                    # End of article
                    if articles_processed == n_articles:
                        break
                    continue

                match_obj = match(pattern, line)
                if match_obj:
                    article_map.append(paragraph_index)
                    assert len(article_map) % 2 == 1
                    # First line of article
                    topics.append(match_obj.group(2).split(' '))
                    articles_processed += 1
                    print(str(articles_processed) + ". article loaded")
                    continue

            paragraph_index += 1
            articles_paragraphs.append(line)

    if len(article_map) != (len(topics) * 2):
        raise ValueError("Error: matrix dimensions do not match")

    return articles_paragraphs, topics, article_map


if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    articles, topics, article_map = build_topics_paragraphs_index_map(config.training_data_path)
    y_true = binarizer.transform(topics)

    # TODO: update the following code

    threshold = 0

    x_list = []
    y_list = []

    # used only for percentage printout
    number_of_articles = len(articles)
    one_percent = int(number_of_articles / 100) if number_of_articles > 100 else 1

    print("Processing the training data")
    for i in range(number_of_articles):
        # vectorize current article, paragraph per matrix row
        x_article = vectorizer.transform(articles[i]['paragraphs'])
        y_dec_article = classifier.decision_function(x_article)

        # set all the topics which were not in the original article to 0
        y_article = y_true[i] * y_dec_article

        x_list.append(x_article)
        y_list.append(y_article)

        if i != 0 and i % one_percent == 0:
            print("{0} % done".format(i / one_percent))

    print("Removing unnecessary variables from memory")
    vectorizer, classifier, corpus, topics, y_true = None, None, None, None, None

    print("Building training matrices")
    # create one matrix from the list of matrices
    x = sp.sparse.vstack(x_list, format='csr')
    x_list = None

    y_raw = sp.vstack(y_list)
    y_list = None

    # set 0 values to -1000000 to make the topics which were not in the original article irrelevant
    y_raw[y_raw == 0] = -1000000
    y = r_cut(y_raw, 1) + (y_raw > threshold)

    classifier_paragraphs = OneVsRestClassifier(LinearSVC(), n_jobs=4)
    print("Training the classifier")
    classifier_paragraphs.fit(x, y)

    print("Saving the classifier to file")
    save_pickle(config.classifier_paragraphs_path, classifier_paragraphs)
