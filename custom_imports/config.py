# coding: utf-8
data_directory = './data'

# Raw data
training_data_raw_path = data_directory + '/CNO_IPTC_train.txt'
testing_data_path = data_directory + '/CNO_IPTC_test.txt'

training_data_path = data_directory + '/training.txt'
held_out_data_path = data_directory + '/held_out.txt'

data_vectorizer_path = data_directory + '/vectorizer.pickle'
topic_binarizer_path = data_directory + '/binarizer.pickle'

classifier_path = data_directory + '/classifier.pickle'

y_paragraphs = data_directory + '/y_paragraphs.npy'
y_paragraphs_true = data_directory + '/y_paragraphs_true.npy'
x_paragraphs = data_directory + '/x_paragraphs.npz'
article_paragraph_map = data_directory + '/article_paragraph_map.pickle'

classifier_paragraphs_path = data_directory + '/classifier_paragraphs.pickle'
