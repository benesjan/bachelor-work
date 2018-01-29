import config
from classification.optimal_threshold_discovery import plot_thresholds
from segmentation.data_preparation import line_map_to_y
from segmentation.distance_based_methods import compute_cosine_distance
from utils import load_pickle, build_topics_paragraphs_index_map

if __name__ == '__main__':
    classifier = load_pickle(config.classifier_binary)
    vectorizer = load_pickle(config.vectorizer)

    paragraphs, ignored, line_map = build_topics_paragraphs_index_map(config.data['held_out'])

    y_true = line_map_to_y(line_map)
    x = vectorizer.transform(paragraphs)

    x_norms = compute_cosine_distance(x)

    y_pred = classifier.decision_function(x_norms)

    plot_thresholds(y_true, y_pred, False, 'binary', False)