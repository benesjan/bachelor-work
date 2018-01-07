# coding: utf-8
data_dir = './data'

# Raw data
train_data_raw = data_dir + '/CNO_IPTC_train.txt'

train_data = data_dir + '/train.txt'
held_out_data = data_dir + '/held_out.txt'
test_data = data_dir + '/CNO_IPTC_test.txt'

vectorizer = data_dir + '/vectorizer.pickle'
binarizer = data_dir + '/binarizer.pickle'
classifier = data_dir + '/classifier.pickle'

# Paragraph classifier data
data_names = ['train', 'held_out', 'test']

par_data_dir = data_dir + '/paragraphs'


def get_par_data(data_name):
    assert data_name in data_names, "Invalid data name"
    recent_dir = par_data_dir + '/' + data_name + '/'
    return {
        'x': recent_dir + 'x.npz',
        'y': recent_dir + 'y.npy',
        'y_true': recent_dir + 'y_true.npy',
        'line_map': recent_dir + 'line_map.pickle'
    }


classifier_par = par_data_dir + '/classifier_par.pickle'
