"""Evaluation Functions

This module contains functions to evaluate the performance of a CWI model.

"""
from sklearn import metrics


def report_binary_score(gold_labels, predicted_labels, detailed=True):
    """Generates a report for the binary classification task.

    The overall performance is measured using macro-F1 score. It is also possible to get label-specific scores.

    Args:
        gold_labels (1d array-like): The gold-standard labels.
        predicted_labels (1d array-like): The predicted labels.
        detailed (bool): If False, only reports the macro-F1 score.
            If True, also reports the per-label precision, recall, F1 and support values.

    Returns:
        str. The report containing the computed scores.

    """
    report_str = ""

    macro_F1 = metrics.f1_score(gold_labels, predicted_labels, average='macro')
    report_str += "macro-F1: {:.2f}".format(macro_F1)
    if detailed:
        scores = metrics.precision_recall_fscore_support(gold_labels, predicted_labels)
        report_str += "\n{:^10}{:^10}{:^10}{:^10}{:^10}".format("Label", "Precision", "Recall", "F1", "Support")
        report_str += '\n' + '-' * 50
        report_str += "\n{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(0, scores[0][0], scores[1][0], scores[2][0], scores[3][0])
        report_str += "\n{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(1, scores[0][1], scores[1][1], scores[2][1], scores[3][1])

    return report_str
