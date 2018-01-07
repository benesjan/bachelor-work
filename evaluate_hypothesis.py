from generate_data_and_instances import build_corpus_and_topics
from custom_imports import config
from custom_imports.classifier_functions import predict_tuned, classifier
from custom_imports.utils import load_pickle, build_topics_and_paragraphs

if __name__ == '__main__':
    vectorizer = load_pickle(config.vectorizer)
    binarizer = load_pickle(config.binarizer)

    corpus, topics = build_corpus_and_topics(config.held_out_data)
    corpus_paragraphs, topics_ = build_topics_and_paragraphs(config.held_out_data)

    x = vectorizer.transform(corpus)
    y = classifier.predict(x)
    y_tuned = predict_tuned(x)

    number_of_articles = len(y)

    fp_articles, fn_articles, fp_articles_tuned, fn_articles_tuned, fp_paragraphs, fn_paragraphs = 0., 0., 0., 0., 0., 0.
    for i in range(0, number_of_articles):
        print("Article index: " + str(i))
        current_topics = set(topics[i])

        y_transformed = set(binarizer.inverse_transform(y[i:i + 1, :])[0])
        y_transformed_tuned = set(binarizer.inverse_transform(y_tuned[i:i + 1, :])[0])

        x_paragraphs = vectorizer.transform(corpus_paragraphs[i]['paragraphs'])
        y_transformed_paragraphs_list = binarizer.inverse_transform(classifier.predict(x_paragraphs))
        y_transformed_paragraphs_set = set()
        y_transformed_paragraphs_set.update(*y_transformed_paragraphs_list)

        fp_articles += len(y_transformed - current_topics)
        fn_articles += len(current_topics - y_transformed)

        fp_articles_tuned += len(y_transformed_tuned - current_topics)
        fn_articles_tuned += len(current_topics - y_transformed_tuned)

        fp_paragraphs += len(y_transformed_paragraphs_set - current_topics)
        fn_paragraphs += len(current_topics - y_transformed_paragraphs_set)

    print("Full articles (per 1 classification) - false positives: %.2f, false negatives: %.2f" % (
        fp_articles / number_of_articles, fn_articles / number_of_articles))

    print("Full articles (per 1 classification) tuned - false positives: %.2f, false negatives: %.2f" % (
        fp_articles_tuned / number_of_articles, fn_articles_tuned / number_of_articles))

    print("Per paragraphs (per 1 classification) - false positives: %.2f, false negatives: %.2f" % (
        fp_paragraphs / number_of_articles, fn_paragraphs / number_of_articles))
