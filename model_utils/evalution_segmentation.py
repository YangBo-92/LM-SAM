from __future__ import division

import numpy as np
import six
import torch
import torch.nn as nn


def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Calculate confusion matrix for semantic segmentation evaluation."""
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 1
    confusion = np.zeros((n_class, n_class), dtype=np.int64)

    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # dynamically expand confusion matrix if new classes appear
        lb_max = np.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion
            n_class = lb_max + 1
            confusion = expanded_confusion

        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class ** 2
        ).reshape((n_class, n_class))

    # ensure both iterators are fully consumed
    for iter_ in (pred_labels, gt_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


def calc_semantic_segmentation_iou(confusion):
    """Compute Intersection over Union (IoU) from confusion matrix."""
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / (iou_denominator + 1e-10)
    return iou[:-1]


def eval_semantic_segmentation(pred_labels, gt_labels, preout, gtout):
    """Evaluate semantic segmentation metrics including IoU, accuracy, and clinical metrics."""
    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)

    pixel_accuracy = np.diag(confusion).sum() / (confusion.sum() + 1e-10)
    class_accuracy = np.diag(confusion) / (confusion.sum(axis=1) + 1e-10)

    return {
        'iou': iou,
        'miou': np.nanmean(iou),
        'pixel_accuracy': pixel_accuracy,
        'class_accuracy': class_accuracy,
        'mean_class_accuracy': np.nanmean(class_accuracy[:-1]),
        'JS': get_JS(preout, gtout),
        'DC': get_dice(preout, gtout),
        'SP': _specificity(preout, gtout),
        'SE': _sensitivity(preout, gtout),
        'PC': _precision(preout, gtout),
        'RE': _recall(preout, gtout),
        'RVD': rvd(preout, gtout),
        'VOE': _VOE(preout, gtout)
    }


def get_JS(preout, gtout):
    """Compute Jaccard Similarity (JS)."""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    intersection = torch.sum((preout + gtout) == 0)
    union = torch.sum((preout + gtout) == 0) + torch.sum((preout + gtout) == 1)

    return float(intersection) / (float(union) + 1e-6)


def get_dice(preout, gtout):
    """Compute Dice Coefficient (DC)."""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    intersection = torch.sum((preout + gtout) == 0)
    union = torch.sum((preout + gtout) <= 1)

    return float(2 * intersection) / (float(union) + 1e-6)


def rvd(preout, gtout):
    """Compute Relative Volume Difference (RVD)."""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    a = torch.sum(preout == 0)
    b = torch.sum(gtout == 0)

    return float(a - b) / (float(b) + 1e-6)


def _VOE(preout, gtout):
    """Compute Volumetric Overlap Error (VOE)."""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    a = torch.sum(preout == 0)
    b = torch.sum(gtout == 0)

    return 2.0 * float(a - b) / (float(a + b) + 1e-6)


def _specificity(preout, gtout):
    """Compute specificity (true negative rate)."""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TN = ((preout == 1) & (gtout == 1))
    FP = ((preout == 0) & (gtout == 1))

    return float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)


def _sensitivity(preout, gtout):
    """Compute sensitivity (true positive rate)."""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TP = ((preout == 0) & (gtout == 0))
    FN = ((preout == 1) & (gtout == 0))

    return float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)


def _precision(preout, gtout):
    """Compute precision (positive predictive value)."""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TP = ((preout == 0) & (gtout == 0))
    FP = ((preout == 0) & (gtout == 1))

    return float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)


def _recall(preout, gtout):
    """Compute recall (same as sensitivity)."""
    return _sensitivity(preout, gtout)


if __name__ == "__main__":
    import torch as t
    import numpy

    # quick sanity check with random data
    pred_labels = numpy.random.randint(6, 256, 256)
    gt_labels = numpy.random.randint(6, 256, 256)
    preout = numpy.random.randint(6, 256, 256)
    gtout = numpy.random.randint(6, 256, 256)