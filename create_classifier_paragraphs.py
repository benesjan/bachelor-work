import scipy as sp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from custom_imports import config
from custom_imports.utils import build_topics_and_paragraphs, load_pickle, r_cut, save_pickle

if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    articles, topics = build_topics_and_paragraphs(config.training_data_path)
    inversed_topics = binarizer.transform(topics)

    print("Removing binarizer from memory")
    binarizer = None

    threshold = 0

    x_list = []
    y_list = []

    number_of_articles = len(articles)
    one_percent = int(number_of_articles / 100) if number_of_articles > 100 else 1

    print("Processing the training data")
    for i in range(number_of_articles):
        x_article = vectorizer.transform(articles[i]['paragraphs'])
        y_dec_article = classifier.decision_function(x_article)
        y_article = inversed_topics[i] * y_dec_article

        x_list.append(x_article)
        y_list.append(y_article)

        if i != 0 and i % one_percent == 0:
            print("{0} % done".format(i / one_percent))

    print("Removing unnecessary variables from memory")
    vectorizer, classifier, corpus, topics, inversed_topics = None, None, None, None, None

    print("Building training matrices")
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
