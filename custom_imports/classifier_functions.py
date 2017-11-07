# coding: utf-8
from custom_imports import config
from custom_imports.utils import load_pickle, r_cut

classifier = load_pickle(config.classifier_path)


def predict_tuned(x):
    print("Classifying the data")
    y_dec = classifier.decision_function(x)

    # Ensures at least 1 predicted topic for each article
    y_pred_min_topics = r_cut(y_dec, 1)

    # Returns matrix where each elements is set to True if the element's value is bigger than threshold
    y_pred_T = y_dec > -0.31  # -0.31 for 1 min topic

    return y_pred_min_topics + y_pred_T
