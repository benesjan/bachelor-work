from sklearn.calibration import CalibratedClassifierCV

from custom_imports.utils import build_topics_and_paragraphs, load_pickle
from custom_imports import config

if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    calibrated_classifier = CalibratedClassifierCV(classifier, cv='prefit')

    articles, topics = build_topics_and_paragraphs(config.testing_data_path, 1)

    for article in articles:
        y_dec_article = calibrated_classifier.predict(vectorizer.transform(article['paragraphs']))
        print(y_dec_article)
