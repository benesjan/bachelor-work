from pathlib import Path
import matplotlib.pyplot as plt

import config
from utils import first_option, load_pickle


def merge_histories(histories):
    merged_dict = histories[0]

    for i in range(1, len(histories)):
        for key in merged_dict.keys():
            merged_dict[key] += histories[i][key]

    return merged_dict


if __name__ == '__main__':
    if first_option('Do you want to use the model trained on cosine distances [c] or on raw SVM predictions [r]?',
                    'c', 'r'):
        cosine = True
        prefix = '1'
    else:
        cosine = False
        prefix = '577'

    files = Path(config.hist_dir).glob('**/history_' + prefix + '*.pickle')
    files = [str(p) for p in files]
    files.sort()

    histories = []
    for file_path in files:
        histories.append(load_pickle(file_path))

    history = merge_histories(histories)

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'held out'], loc='lower right')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'held out'], loc='upper right')

    plt.show()
