from custom_imports import config
from custom_imports.utils import build_topics_and_paragraphs, load_pickle, r_cut
import numpy as np

if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    articles, topics = build_topics_and_paragraphs(config.training_data_path, 1)
    inversed_topics = binarizer.transform(topics)

    threshold = 0

    for i in range(len(articles)):
        article_topics_indices = np.where(inversed_topics[i] == 1)[0]
        y_dec_article = classifier.decision_function(vectorizer.transform(articles[i]['paragraphs']))
        topic_values = y_dec_article[:, article_topics_indices]
        y = r_cut(topic_values, 1) + topic_values > threshold
        print(binarizer.transform(topics[i]))
