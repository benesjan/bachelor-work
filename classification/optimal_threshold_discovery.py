# coding: utf-8
import config
from classification.create_classifier_paragraphs import threshold_half_max, process_y, threshold_biggest_gap
from utils import load_pickle, build_corpus_and_topics, load_sparse_csr, first_option, plot_thresholds

if __name__ == '__main__':

    if first_option('Do you want to use paragraphs trained classifier [p] or the article trained version? [a]',
                    'p', 'a'):
        data = config.get_par_data('held_out')

        if first_option('Do you want to use the biggest gap thresholding mechanism [b]'
                        ' or half the biggest probability as threshold [h]?', 'b', 'h'):
            print('Loading the paragraph trained classifier trained on data processed by'
                  'biggest gap thresholding mechanism ')
            classifier = load_pickle(config.classifier_par_biggest_gap)
            y_true = process_y(data, threshold_biggest_gap)
        else:
            print('Loading the paragraph trained classifier trained on data processed by'
                  ' threshold_half_max function')
            classifier = load_pickle(config.classifier_par_half_max)
            y_true = process_y(data, threshold_half_max)

        print("Loading x")
        x = load_sparse_csr(data['x'])
    else:
        print('Loading the full article trained classifier')
        vectorizer = load_pickle(config.vectorizer)
        binarizer = load_pickle(config.binarizer)

        classifier = load_pickle(config.classifier)

        corpus, topics = build_corpus_and_topics(config.data['held_out'])

        print("Transforming corpus by vectorizer")
        x = vectorizer.transform(corpus)
        print("Transforming article topics by binarizer")
        y_true = binarizer.transform(topics)
        del vectorizer, binarizer, corpus, topics

    print("Classifying the data")
    y_pred = classifier.predict_proba(x)

    plot_thresholds(y_true, y_pred, True)
