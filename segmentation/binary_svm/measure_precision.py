import config
from segmentation.data_preparation import line_map_to_y
from segmentation.distance_based_methods import compute_cosine_distance
from utils import load_pickle, build_topics_paragraphs_index_map
from sklearn.metrics import precision_recall_fscore_support as prfs

if __name__ == '__main__':
    classifier = load_pickle(config.classifier_binary)
    vectorizer = load_pickle(config.vectorizer)

    paragraphs, ignored, line_map = build_topics_paragraphs_index_map(config.data['test'])

    y_true = line_map_to_y(line_map)
    x = vectorizer.transform(paragraphs)

    x_norms = compute_cosine_distance(x)

    y_pred = classifier.decision_function(x_norms) > -0.55

    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
