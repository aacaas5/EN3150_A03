"""Explicit macro averaging over all ten original dataset classes."""
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def classification_metrics(truth, predicted, classes):
    labels = list(range(len(classes)))
    metrics = {"test_accuracy": accuracy_score(truth, predicted),
               "macro_precision": precision_score(truth, predicted, labels=labels, average="macro", zero_division=0),
               "macro_recall": recall_score(truth, predicted, labels=labels, average="macro", zero_division=0)}
    matrix = confusion_matrix(truth, predicted, labels=labels)
    report = classification_report(truth, predicted, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    return metrics, matrix, report
