import numpy as np

import config
from utils import print_measurements

if __name__ == '__main__':
    data = config.get_seg_data('test')

    print("Loading the data")
    y_true = np.load(data['y_true_lm'])

    y_pred = np.random.choice(a=[False, True], size=y_true.shape)

    print_measurements(y_true, y_pred)
