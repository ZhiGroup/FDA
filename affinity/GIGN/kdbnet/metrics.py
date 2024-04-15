from sklearn import metrics
from scipy import stats
import numpy as np

def eval_mse(y_true, y_pred, squared=True):
    """Evaluate mse/rmse and return the results.
    squared: bool, default=True
        If True returns MSE value, if False returns RMSE value.
    """
    return metrics.mean_squared_error(y_true, y_pred, squared=squared)

def eval_pearson(y_true, y_pred):
    """Evaluate Pearson correlation and return the results."""
    return stats.pearsonr(y_true, y_pred)[0]

def eval_spearman(y_true, y_pred):
    """Evaluate Spearman correlation and return the results."""
    return stats.spearmanr(y_true, y_pred)[0]

def eval_r2(y_true, y_pred):
    """Evaluate R2 and return the results."""
    return metrics.r2_score(y_true, y_pred)

def eval_auroc(y_true, y_pred):
    """Evaluate AUROC and return the results."""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)

def eval_auprc(y_true, y_pred):
    """Evaluate AUPRC and return the results."""
    pre, rec, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(rec, pre)


def evaluation_metrics(y_true=None, y_pred=None,
		eval_metrics=[]):
    """Evaluate eval_metrics and return the results.
    Parameters
    ----------
    y_true: true labels
    y_pred: predicted labels
    eval_metrics: a list of evaluation metrics
    """
    results = {}
    for m in eval_metrics:
        if m == 'mse':
            s = eval_mse(y_true, y_pred, squared=True)
        elif m == 'rmse':
            s = eval_mse(y_true, y_pred, squared=False)
        elif m == 'pearson':
            s = eval_pearson(y_true, y_pred)
        elif m == 'spearman':
            s = eval_spearman(y_true, y_pred)
        elif m == 'r2':
            s = eval_r2(y_true, y_pred)
        elif m == 'auroc':
            s = eval_auroc(y_true, y_pred)
        elif m == 'auprc':
            s = eval_auprc(y_true, y_pred)
        else:
            raise ValueError('Unknown evaluation metric: {}'.format(m))
        results[m] = s        
    return results