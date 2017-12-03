from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from custom_imports import config
from custom_imports.utils import save_pickle, build_corpus_and_topics


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

    print("Building the data matrix using the TfidfVectorizer")
    X = vectorizer.fit_transform(corpus)

    print("Building the label matrix by MultiLabelBinarizer")
    Y = binarizer.fit_transform(topics)

    print("Saving the vectorizer to file")
    save_pickle(config.data_vectorizer_path, vectorizer)

    print("Saving the binarizer to file")
    save_pickle(config.topic_binarizer_path, binarizer)

    print("Removing unnecessary variables from memory")
    vectorizer, binarizer, corpus, topics = None, None, None, None

    classifier = OneVsRestClassifier(LinearSVC(), n_jobs=4)
    print("Training the classifier")
    classifier.fit(X, Y)

    print("Saving the classifier to file")
    save_pickle(config.classifier_path, classifier)
