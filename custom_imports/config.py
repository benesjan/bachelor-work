# coding: utf-8
data_dir = './data'

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
    return {
        'dir': recent_dir,
        'name': data_name,
        'text': data[data_name],
        'x': recent_dir + 'x.npz',
        'line_map': recent_dir + 'line_map.pickle'
    }