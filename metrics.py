import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def mosei_metrics(y_true, y_pred):
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
    y_true_bin = y_true[non_zeros] > 0
    y_pred_bin = y_pred[non_zeros] > 0

    y_pred_a7 = np.clip(y_pred, a_min=-3., a_max=3.)
    y_true_a7 = np.clip(y_true, a_min=-3., a_max=3.)
    y_pred_a5 = np.clip(y_pred, a_min=-2., a_max=2.)
    y_true_a5 = np.clip(y_true, a_min=-2., a_max=2.)

    mult_a7 = multiclass_acc(y_pred_a7, y_true_a7)
    mult_a5 = multiclass_acc(y_pred_a5, y_true_a5)
    bi_acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, average='weighted')
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_pred.reshape(-1), y_true.reshape(-1))[0][1]
    return mult_a7, mult_a5, bi_acc, f1, mae, corr


def mosei_metrics_with_zero(y_true, y_pred, data_type="mosei"):
    if data_type == "sims":
        y_true_bin = y_true > 0
        y_pred_bin = y_pred > 0
    else:
        y_true_bin = y_true >= 0
        y_pred_bin = y_pred >= 0

    y_pred_a7 = np.clip(y_pred, a_min=-3., a_max=3.)
    y_true_a7 = np.clip(y_true, a_min=-3., a_max=3.)
    y_pred_a5 = np.clip(y_pred, a_min=-2., a_max=2.)
    y_true_a5 = np.clip(y_true, a_min=-2., a_max=2.)

    mult_a7 = multiclass_acc(y_pred_a7, y_true_a7)
    mult_a5 = multiclass_acc(y_pred_a5, y_true_a5)
    bi_acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, average='weighted')
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_pred.reshape(-1), y_true.reshape(-1))[0][1]
    return mult_a7, mult_a5, bi_acc, f1, mae, corr


METRICS = {
    "mosei": mosei_metrics
}