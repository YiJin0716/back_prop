"""Threshold ROC uses continuous probabilities, never thresholded predictions."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def binary_auc(labels, scores):
    labels, scores = np.asarray(labels,float), np.asarray(scores,float)
    keep = np.isfinite(labels) & np.isfinite(scores)
    y, p = labels[keep], scores[keep]
    if not set(np.unique(y)).issubset({0.,1.}):
        raise ValueError('AUC requires binary target labels')
    result = dict(n_total=len(labels), n_evaluated=int(keep.sum()),
                  n_excluded=int((~keep).sum()), n_positive=int((y==1).sum()),
                  n_negative=int((y==0).sum()))
    if len(np.unique(y))<2:
        return dict(result,auc=None,status='undefined_single_class'),None
    fpr,tpr,threshold=roc_curve(y,p,drop_intermediate=False)
    return dict(result,auc=float(roc_auc_score(y,p)),status='ok'),(fpr,tpr,threshold)


def threshold_auc(ratings, probabilities, threshold, rule='ge'):
    ratings, probabilities = np.asarray(ratings),np.asarray(probabilities,float)
    if probabilities.shape != (len(ratings),5) or threshold not in range(1,6):
        raise ValueError('Expected five class probabilities and threshold 1..5')
    if rule not in ('ge','gt') or not np.isin(ratings,[1,2,3,4,5]).all():
        raise ValueError('Invalid threshold rule or official ordinal ratings')
    valid=np.isfinite(probabilities).all(1)
    if valid.any() and (np.any(probabilities[valid]<0) or
                       not np.allclose(probabilities[valid].sum(1),1,atol=1e-5)):
        raise ValueError('Invalid predicted class probabilities')
    offset=threshold-1 if rule=='ge' else threshold
    score=probabilities[:,offset:].sum(1)
    score[~valid]=np.nan  # also needed when the selected class slice is empty
    target=ratings>=threshold if rule=='ge' else ratings>threshold
    return binary_auc(target.astype(int),score)
