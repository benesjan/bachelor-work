data_directory = './data'

# Raw data
training_data_path = data_directory + '/CNO_IPTC_train.txt'
testing_data_path = data_directory + '/CNO_IPTC_test.txt'

# Data processed by vectorizer and binarizer
data_matrix_path = data_directory + '/matrix.npz'
topics_matrix_path = data_directory + '/topics.npy'

data_vectorizer_path = data_directory + '/vectorizer.pickle'
topic_binarizer_path = data_directory + '/binarizer.pickle'

classifier_path = data_directory + '/classifier.pickle'
