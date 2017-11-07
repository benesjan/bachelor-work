from generate_data_and_instances import build_corpus_and_topics
from custom_imports import config
from custom_imports.classifier_functions import predict_tuned
from custom_imports.utils import load_pickle, build_topics_and_paragraphs

if __name__ == '__main__':
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    corpus, topics = build_corpus_and_topics(config.held_out_data_path)
    corpus_paragraphs, topics_ = build_topics_and_paragraphs(config.held_out_data_path)

    x = vectorizer.transform(corpus)
    y = predict_tuned(x)

    number_of_articles = len(y)

    fp_articles, fn_articles, fp_paragraphs, fn_paragraphs = 0., 0., 0., 0.
    for i in range(0, number_of_articles):
        print("Article index: " + str(i))
        y_transformed = set(binarizer.inverse_transform(y[i:i + 1, :])[0])
        current_topics = set(topics[i])

        x_paragraphs = vectorizer.transform(corpus_paragraphs[i]['paragraphs'])
        y_transformed_paragraphs_list = binarizer.inverse_transform(predict_tuned(x_paragraphs))
        y_transformed_paragraphs_set = set()
        y_transformed_paragraphs_set.update(*y_transformed_paragraphs_list)

        fp_articles += len(y_transformed - current_topics)
        fn_articles += len(current_topics - y_transformed)

        fp_paragraphs += len(y_transformed_paragraphs_set - current_topics)
        fn_paragraphs += len(current_topics - y_transformed_paragraphs_set)

    print("Full articles (per 1 classification) - false positives: %.2f, false negatives: %.2f" % (
        fp_articles / number_of_articles, fn_articles / number_of_articles))

    print("Per paragraphs (per 1 classification) - false positives: %.2f, false negatives: %.2f" % (
        fp_paragraphs / number_of_articles, fn_paragraphs / number_of_articles))
