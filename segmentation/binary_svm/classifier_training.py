from sklearn.svm import LinearSVC

import config
from segmentation.data_preparation import line_map_to_y
from segmentation.distance_based_methods import compute_cosine_distance
from utils import build_topics_paragraphs_index_map, load_pickle, save_pickle

if __name__ == '__main__':
    vectorizer = load_pickle(config.vectorizer)

    paragraphs, ignored, line_map = build_topics_paragraphs_index_map(config.data['train'])

    y_true = line_map_to_y(line_map)
    x = vectorizer.transform(paragraphs)

    x_norms = compute_cosine_distance(x)

    del paragraphs, ignored, line_map, vectorizer, x

    classifier = LinearSVC(random_state=0)
    classifier.fit(x_norms, y_true)

    save_pickle(config.classifier_binary, classifier)
