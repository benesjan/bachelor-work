import config
from data_processor import build_corpus_and_topics
from data_utils import load_pickle
from precision_measurement_paragraphs import build_topics_and_paragraphs

if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    number_of_articles = 10
    corpus, topics = build_corpus_and_topics(config.testing_data_path, number_of_articles)
    corpus_paragraphs, topics_ = build_topics_and_paragraphs(config.testing_data_path, number_of_articles)

    x = vectorizer.transform(corpus)
    y = classifier.predict(x)

    for i in range(0, len(y)):
        y_transformed = set(binarizer.inverse_transform(y[i:i + 1, :])[0])
        current_topics = set(topics[i])
        false_positives = y_transformed - current_topics
        false_negatives = current_topics - y_transformed
        if not false_positives or not false_negatives:
            # Exclude perfect predictions
            continue

        print("\n-----------------\nActual: " + str(current_topics))
        print("Predicted - full article: " + str(y_transformed))
        x_paragraphs = vectorizer.transform(corpus_paragraphs[i]['paragraphs'])
        y_transformed_paragraphs_list = binarizer.inverse_transform(classifier.predict(x_paragraphs))
        print("Predicted - paragraphs: " + str(y_transformed_paragraphs_list) + "\n")
        y_transformed_paragraphs_set = set()
        y_transformed_paragraphs_set.update(*y_transformed_paragraphs_list)

        print("False positives:")
        for false_positive in false_positives:
            print(false_positive + (" - IN" if false_positive in y_transformed_paragraphs_set else " - NOT in")
                  + " paragraphs")

        if not false_negatives:
            continue

        print("\nFalse negatives: ")
        for false_negative in false_negatives:
            print(false_negative + (" - IN" if false_positive in y_transformed_paragraphs_set else " - NOT in")
                  + " paragraphs")
