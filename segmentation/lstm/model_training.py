from pathlib import Path
import numpy as np
from keras.models import load_model

import config
from segmentation.lstm.lstm_utils import split_to_time_steps, shuffle_the_data, build_model
from utils import first_option, create_dir, save_pickle

if __name__ == '__main__':
    # Number of vectors in one sequence, input data structure [samples, time_steps, features]
    time_steps = 200

    if Path(config.lstm_model_577).is_file() and first_option('Do you want to continue training the saved model?',
                                                              'y', 'n'):
        print("Loading new model")
        model = load_model(config.lstm_model_577)
    else:
        print("Building new model")
        model = build_model(time_steps, 577)

    # Create dir for histories
    create_dir(config.hist_dir)

    print("Loading the data")
    train = config.get_seg_data('train')
    held_out = config.get_seg_data('held_out')

    X_train_or = np.load(train['y'])
    y_train_or = np.load(train['y_true_lm'])

    X_held_out = np.load(held_out['y'])
    y_held_out = np.load(held_out['y_true_lm'])

    # Split the 2D matrix to 3D matrix of dimensions [samples, time_steps, features]
    X_held_out = split_to_time_steps(X_held_out)

    # Append 0 value to the beginning of y so the values represent if there was a boundary between current sample and
    # the previous one, not the current and next
    y_train_or = np.append(0, y_train_or)
    y_held_out = np.append(0, y_held_out)

    # Split the 1D vector to 2D matrix of dimensions: [samples, time_steps]
    y_held_out = split_to_time_steps(y_held_out)

    y_held_out = np.reshape(y_held_out, (y_held_out.shape[0], y_held_out.shape[1], 1))

    shuffling_epochs = 2
    for i in range(shuffling_epochs):
        print("Shuffling epoch " + str(i) + "/" + str(shuffling_epochs))

        [X_train, y_train] = shuffle_the_data(X_train_or, y_train_or)

        X_train = split_to_time_steps(X_train)
        y_train = split_to_time_steps(y_train)
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

        history = model.fit(X_train, y_train, validation_data=(X_held_out, y_held_out), epochs=100, batch_size=100,
                            shuffle=False)

        save_pickle(config.hist_dir + "/history_577_" + str(i) + ".pickle", history.history)

    model.save(config.lstm_model_577)
