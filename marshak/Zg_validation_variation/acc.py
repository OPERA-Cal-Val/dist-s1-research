import numpy as np


def get_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return tp, fp, fn, tn


def get_f1_score(y_true, y_pred):
    tp, fp, fn, _ = get_confusion_matrix(y_true, y_pred)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1


def get_user_accuracy(y_true, y_pred):
    tp, fp, _, _ = get_confusion_matrix(y_true, y_pred)
    user_accuracy = tp / (tp + fp)
    return user_accuracy


def get_producer_accuracy(y_true, y_pred):
    tp, _, fn, _ = get_confusion_matrix(y_true, y_pred)
    producer_accuracy = tp / (tp + fn)
    return producer_accuracy


def get_total_accuracy(y_true, y_pred):
    tp, fp, fn, tn = get_confusion_matrix(y_true, y_pred)
    total_accuracy = (tp + tn) / (tp + tn + fp + fn)
    return total_accuracy
