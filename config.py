import os

# coding: utf-8
data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data'

# Raw data
train_data_raw = data_dir + '/CNO_IPTC_train.txt'

data = {
    'train': data_dir + '/train.txt',
    'held_out': data_dir + '/held_out.txt',
    'test': data_dir + '/CNO_IPTC_test.txt'
}

vectorizer = data_dir + '/vectorizer.pickle'
binarizer = data_dir + '/binarizer.pickle'
classifier = data_dir + '/classifier.pickle'

# Paragraph classifier data
par_data_dir = data_dir + '/paragraphs'


def get_par_data(data_name):
    assert data_name in data.keys(), "Invalid data name"
    recent_dir = par_data_dir + '/' + data_name + '/'
    return {
        'dir': recent_dir,
        'name': data_name,
        'text': data[data_name],
        'x': recent_dir + 'x.npz',
        'y': recent_dir + 'y.npy',
        'y_true': recent_dir + 'y_true.npy',
        'line_map': recent_dir + 'line_map.pickle'
    }


classifier_par_half_max = par_data_dir + '/classifier_par_half_max.pickle'
classifier_par_biggest_gap = par_data_dir + '/classifier_par_biggest_gap.pickle'

# Text segmentation data
seg_data_dir = data_dir + '/segmentation'


def get_seg_data(data_name):
    assert data_name in data.keys(), "Invalid data name"
    recent_dir = seg_data_dir + '/' + data_name + '/'
    par_data = get_par_data(data_name)
    return {
        'dir': recent_dir,
        'name': data_name,
        'text': data[data_name],
        'x': par_data['x'],
        'y': par_data['y'],
        'y_true_lm': recent_dir + 'y_true_lm.npy',
        'line_map': par_data['line_map']
    }


classifier_linear = seg_data_dir + '/classifier_linear.pickle'
classifier_rbf = seg_data_dir + '/classifier_rbf.pickle'

lstm_model_1 = seg_data_dir + '/lstm_model_1.h5'
lstm_model_577 = seg_data_dir + '/lstm_model_577.h5'

hist_dir = seg_data_dir + '/lstm_histories'

# Live subtitles
subtitles_path = data_dir + '/udalosti/txt'
