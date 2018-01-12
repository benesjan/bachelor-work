from sklearn.calibration import CalibratedClassifierCV
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
            open(held_out_path, 'w', encoding='utf-8') as held_out, \
            open(training_data_path, 'w', encoding='utf-8') as training:
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
    separate_data(config.train_data_raw, config.data['held_out'], config.data['train'])

    corpus, topics = build_corpus_and_topics(config.data['train'])

    print("Building the data matrix using the TfidfVectorizer")
    X = vectorizer.fit_transform(corpus)

    print("Building the label matrix by MultiLabelBinarizer")
    Y = binarizer.fit_transform(topics)

    print("Saving the vectorizer to file")
    save_pickle(config.vectorizer, vectorizer)

    print("Saving the binarizer to file")
    save_pickle(config.binarizer, binarizer)

    print("Removing unnecessary variables from memory")
    del vectorizer, binarizer, corpus, topics

    classifier_one_class = CalibratedClassifierCV(LinearSVC(), cv=3)
    classifier = OneVsRestClassifier(classifier_one_class, n_jobs=1)

    print("Training the classifier")
    classifier.fit(X, Y)

    print("Saving the classifier to file")
    save_pickle(config.classifier, classifier)
