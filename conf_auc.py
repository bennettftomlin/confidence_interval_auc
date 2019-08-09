# -*- coding: utf-8 -*-
import numpy as np
import sklearn
from sklearn import metrics


def conf_auc(test_predictions, ground_truth, bootstrap=1000, seed=None,  confint=0.95):
    """Takes as input test predictions, ground truth, number of bootstraps, seed, and confidence interval"""
    #inspired by https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals by ogrisel
    bootstrapped_scores = []
    rng = np.random.RandomState(seed)
    if confint>1:
        confint=confint/100
    for i in range(bootstrap):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(test_predictions) - 1, len(test_predictions))
        if len(np.unique(ground_truth[indices])) < 2:
            continue

        score = metrics.roc_auc_score(ground_truth[indices], test_predictions[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower_bound=(1-confint)/2
    upper_bound=1-lower_bound
    confidence_lower = sorted_scores[int(lower_bound * len(sorted_scores))]
    confidence_upper = sorted_scores[int(upper_bound * len(sorted_scores))]
    print("{:0.0f}% confidence interval for the score: [{:0.3f} - {:0.3}]".format(confint*100, confidence_lower, confidence_upper))
    confidence_interval = (confidence_lower, confidence_upper)
    return confidence_interval

