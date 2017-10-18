import re

import config
from data_utils import load_pickle


def build_topics_and_vectorized_paragraphs(raw_data_file_path, vectorizer, n_articles=1):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'
    articles, topics = [], []
    current_article = []

    articles_processed = 0
    with open(raw_data_file_path, 'r', encoding='utf-8') as handler:
        for line in handler:
            if line.startswith('<'):
                if line == '</article>\n':
                    articles.append(current_article)
                    current_article.clear()
                    if articles_processed == n_articles:
                        break
                    continue

                match_obj = re.match(pattern, line)
                if match_obj:
                    topics.append(match_obj.group(2).split(' '))
                    articles_processed += 1
                    if current_article:
                        print("Warning: Non empty article string")
                    print(str(articles_processed) + ". article loaded")
                    continue

            print(line)
            print(vectorizer.transform(line))
            current_article.append(vectorizer.transform(line))

    if len(articles) != len(topics):
        raise ValueError("Error: matrix dimensions do not match")

    return articles, topics


if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    articles, topics = build_topics_and_vectorized_paragraphs(config.testing_data_path, vectorizer)

    article_topics = []
    for article in articles:
        for paragraph in article:
            print(binarizer.inverse_transform(classifier.predict(paragraph)))

    print("Actual topics: " + str(topics))
