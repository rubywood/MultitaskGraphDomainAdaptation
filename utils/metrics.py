import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from scipy.stats import pearsonr
import torch
import torch.nn as nn


def calc_metrics(targets, outputs, cutoff=None):
    if cutoff:
        predictions = np.where(outputs > cutoff, 1, 0)
    else:
        predictions = np.round(outputs)

    print('Targets:', targets)
    print('Outputs:', outputs)
    # slide-level accuracy and AUC - should be same as saved best from training/validation
    slide_level_auc = roc_auc_score(targets, outputs, average='weighted')
    slide_level_acc = accuracy_score(targets, predictions)
    weighted_acc = balanced_accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='weighted')
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')

    print('- Weighted AUC', slide_level_auc)
    print('- Accuracy', slide_level_acc)
    print('- Balanced accuracy', weighted_acc)
    print('- Weighted F1 score', f1)
    print('- Weighted Precision', precision)
    print('- Weighted Recall', recall)
    print()

    return slide_level_auc, slide_level_acc, weighted_acc, f1, precision, recall


def create_resp_metric_dict(resp, true, prob, best_epoch, cutoff=None):
    _, acc, balanced_acc, w_f1, w_precision, w_recall = calc_metrics(true, prob, cutoff)
    try:
        auc = auroc_scorer(true, prob)
        auprc = auprc_scorer(true, prob)
    except ValueError as e:
        print(f"Couldn't calculate metrics due to error: {e}")
        auc, auprc = None, None

    return {f"{resp}-auroc": auc, f"{resp}-auprc": auprc, f'{resp}-acc': acc,
            f'{resp}-balanced_acc': balanced_acc, f'{resp}-weighted_f1': w_f1,
            f'{resp}-weighted_precision': w_precision,
            f'{resp}-weighted_recall': w_recall, f'best_epoch': best_epoch}


def create_multiclass_resp_metric_dict(resp, true, logits, best_epoch):
    probs = nn.functional.softmax(torch.Tensor(logits), dim=1).numpy()
    pred = np.argmax(probs, axis=1)

    print('Targets:', true)
    #print('Probs:', probs)
    print('Outputs:', pred)

    slide_level_auc = auroc_scorer(true, probs, average='macro', multi_class='ovr')
    slide_level_acc = accuracy_score(true, pred)
    weighted_acc = balanced_accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='weighted', labels=[0, 1, 2, 3])
    precision = precision_score(true, pred, average='weighted', labels=[0, 1, 2, 3])
    recall = recall_score(true, pred, average='weighted', labels=[0, 1, 2, 3])

    print('- Weighted AUC', slide_level_auc)
    print('- Accuracy', slide_level_acc)
    print('- Balanced accuracy', weighted_acc)
    print('- Weighted F1 score', f1)
    print('- Weighted Precision', precision)
    print('- Weighted Recall', recall)
    print()

    return {f"{resp}-auroc": slide_level_auc, f'{resp}-acc': slide_level_acc,
            f'{resp}-balanced_acc': weighted_acc, f'{resp}-weighted_f1': f1,
            f'{resp}-weighted_precision': precision,
            f'{resp}-weighted_recall': recall, f'best_epoch': best_epoch}


def continuous_resp_metric_dict(resp, true, prob, best_epoch):
    pearson, _ = pearsonr(true, prob)
    mae = nn.L1Loss()(torch.Tensor(prob), torch.Tensor(true)).item()
    print('- Pearson', pearson)
    print('- MAE', mae)
    print()
    
    return {f"{resp}-pearson": pearson, f"{resp}-mae": mae, f'best_epoch': best_epoch}


def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def threshold_predictions(target, predicted):
    threshold = find_optimal_cutoff(target, predicted)
    return list(map(lambda x: 1 if x > threshold else 0, predicted))


def metric_str_thresh(resp_mets, resp, metric, threshold=False):
    threshold_str = f'threshold-' if threshold else ''
    return f' {resp_mets[f"{threshold_str}{resp}-{metric}"].item():.3f} |'


def metric_str_thresh_all(resp_mets_all, responses, metric, threshold=False):
    met_str = ''
    for i in range(len(resp_mets_all)):
        met_str += metric_str_thresh(resp_mets_all[i], responses[i], metric, threshold)
    return met_str
