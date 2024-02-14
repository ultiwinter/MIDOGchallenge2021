from evalutils.scorers import score_detection, DetectionScore
from collections import namedtuple


def get_confusion_matrix(ground_truth, prediction):
    """ Provides F1 score, recall and precision for binary detection problems."""
    """ We take 30 pixels, which corresponds to approx. 7.5 microns """

    if hasattr(ground_truth, "dim") and ground_truth.dim() == 2:
        sc = score_detection(ground_truth=ground_truth, predictions=prediction, radius=30)
        tp = sc.true_positives
        fp = sc.false_positives
        fn = sc.false_negatives
    else:
        tp, fp, fn = 0, 0, 0
        for gt, pred in zip(ground_truth, prediction):
            sc = score_detection(ground_truth=gt, predictions=pred, radius=30)
            tp += sc.true_positives
            fp += sc.false_positives
            fn += sc.false_negatives
    return tp, fp, fn


def get_metrics(tp, fp, fn):
          
    aggregate_results = dict()

    aggregate_results["precision"] = tp / (tp + fp + 1e-7)
    aggregate_results["recall"] = tp / (tp + fn + + 1e-7)
    aggregate_results["f1_score"] = 2 * tp / ((2 * tp) + fp + fn + 1e-7)

    return aggregate_results
