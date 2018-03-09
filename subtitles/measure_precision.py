from pathlib import Path
import numpy as np
from keras.models import load_model
from matplotlib import pyplot, rc
from nltk import windowdiff, pk
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics.pairwise import cosine_distances

import config
from segmentation.distance_based_methods import compute_distance
from segmentation.lstm.lstm_utils import split_to_time_steps
from utils import load_pickle, first_option


def process_data(path, window=10):
    files = Path(path).glob('**/*.txt')
    X, y = [], []

    leftover = ''
    for file_path in files:
        article = leftover
        with open(str(file_path), 'r', encoding='cp1250') as f:
            for line in f:
                article += line

        # Split text to chunks
        article_list = article.split()
        while len(article_list) != 0:
            list_selection = article_list[0:window]
            if len(list_selection) == window:
                X.append(" ".join(list_selection))
                if leftover:
                    y.append(1)
                    leftover = ''
                else:
                    y.append(0)

                article_list = article_list[window:]
            else:
                leftover = " ".join(article_list)
                article_list = []

        if not leftover:
            y[-1] = 1

    if leftover:
        X.append(leftover)

    y = np.array(y)
    return [X, y]


def plot_thresholds_pk(y_true, y_pred, interval=(0, 1)):
    threshold_array = np.arange(interval[0], interval[1], 0.01)

    values = np.ones((threshold_array.shape[0], 2), dtype=np.float)

    optimal_position = [-100, 0]

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    for i, T in enumerate(threshold_array):
        y_thresholded = y_pred > T

        y_true_joined = "".join(["" + str(x) for x in y_true])
        y_pred_joined = "".join(["" + "1" if x else "0" for x in y_thresholded])

        Pk = pk(y_true_joined, y_pred_joined, boundary="1")

        values[i, :] = [T, Pk]

        print('threshold = %.2f, Pk = %.3f' % (T, Pk))

        if P > optimal_position[1]:
            optimal_position = values[i, :]

    rc('font', family='Arial')
    pyplot.plot(values[:, 0], values[:, 1:4])
    pyplot.legend(['Pk'])

    pyplot.grid()
    pyplot.scatter(optimal_position[0], optimal_position[1], marker="x", s=300, linewidth=3.3)
    pyplot.annotate('[%.2f, %.2f]' % (optimal_position[0], optimal_position[1]),
                    [optimal_position[0], optimal_position[1] - 0.055])

    pyplot.xlim(interval)
    pyplot.ylim([0, 1])

    pyplot.title('Vývoj Pk v závislosti na prahu')
    pyplot.xlabel('Práh')

    pyplot.show()


if __name__ == '__main__':
    [corpus, y_true] = process_data(config.subtitles_path, window=20)

    vectorizer = load_pickle(config.vectorizer)

    print('Transforming corpus by vectorizer')
    corpus_tfidf = vectorizer.transform(corpus)

    print('Loading the classifier')
    classifier = load_pickle(config.classifier)

    X = classifier.predict_proba(corpus_tfidf)

    del corpus, corpus_tfidf, classifier, vectorizer

    # LSTM part
    if first_option('Do you want to use the model trained on cosine distances [c] or on raw SVM predictions [r]?',
                    'c', 'r'):
        print("Computing the distances")
        X = compute_distance(X, cosine_distances)
        model = load_model(config.lstm_model_1)
        T = 0.41
        # y_true = y_true[1:]
        assert len(X) == len(y_true), "Dimensions do not match: y.shape = " + str(y_true.shape) + " X.shape = " + str(
            X.shape)
    else:
        cosine = False
        model = load_model(config.lstm_model_577)
        T = 0.59

    time_steps = model.get_config()[0]['config']['batch_input_shape'][1]

    X = split_to_time_steps(X)
    y_true = split_to_time_steps(y_true)

    y_pred = model.predict(X) > T

    P, R, F, S = prfs(y_true.flatten(), y_pred.flatten(), average='binary')
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))

    # plot_thresholds_pk(y_true, y_pred)

    y_true_joined = "".join(["" + str(x) for x in y_true.flatten()])
    y_pred_joined = "".join(["" + "1" if x else "0" for x in y_pred.flatten()])
    wd = windowdiff(y_true_joined, y_pred_joined, k=100, boundary="1")  # TODO: optimize k
    pk_m = pk(y_true_joined, y_pred_joined, boundary="1")

    print("WindowDiff = " + str(wd) + ", Pk = " + str(pk_m))
