import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense


def split_to_time_steps(x, steps=200):
    """
    Processes the data so that they are suitable for LSTMs
    :param x: 2D matrix
    :param steps: number of steps in the time sequence
    :return: 3D matrix with the following dimensions: [samples, time_steps, features]
    """
    n = int(len(x) / steps) * steps
    x_list = list()
    for i in range(0, n, steps):
        sample = x[i:i + steps]
        x_list.append(sample)
    return np.array(x_list)


def shuffle_the_data(X, y):
    y_shuffled = np.zeros(y.shape, dtype=int)

    # Ensures that the last article is not omitted
    y = np.append(y, 1)
    X_articles = []
    article_start = 0
    for i, val in enumerate(y):
        if val == 1:
            X_articles.append(X[article_start:i])
            article_start = i

    np.random.shuffle(X_articles)

    article_end = 0
    for article in X_articles:
        article_end += len(article)
        if article_end != y_shuffled.shape[0]:
            y_shuffled[article_end] = 1

    X_shuffled = np.vstack(X_articles)
    return [X_shuffled, y_shuffled]


def build_model(time_steps, n_features):
    model = Sequential()

    # stateful=True means that the state is propagated to the next batch
    model.add(LSTM(32, input_shape=(time_steps, n_features), stateful=False, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
