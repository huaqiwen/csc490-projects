from dataclasses import dataclass
from re import A
from typing import List

import torch

from detection.metrics.types import EvaluationFrame


@dataclass
class PRCurve:
    """A precision/recall curve.

    Attributes:
        precision: [N] vector of precision values, where N is the total number of detections.
            The element at index n denotes the precision of the top n detections when ordered by
            decreasing detection scores.
        recall: [N] vector of recall values, where N is the total number of detections.
            The element at index n denotes the recall of the top n detections when ordered by
            decreasing detection scores.
    """

    precision: torch.Tensor
    recall: torch.Tensor


@dataclass
class AveragePrecisionMetric:
    """Stores average precision and its associate precision-recall curve."""

    ap: float
    pr_curve: PRCurve


def compute_precision_recall_curve(
    frames: List[EvaluationFrame], threshold: float
) -> PRCurve:
    """Compute a precision/recall curve over a batch of evaluation frames.

    The PR curve plots the trade-off between precision and recall when sweeping
    across different score thresholds for your detections. To compute precision
    and recall for a score threshold s_i, consider the set of detections with
    scores greater than or equal to s_i. A detection is a true positive if it
    matches a ground truth label; it is a false positive if it does not.

    With this, we define precision = TP / (TP + FP) and recall = TP / (TP + FN),
    where TP is the number of true positive detections, FP is the number of false
    positive detections, and FN is the number of false negative labels (i.e. the
    number of ground truth labels that did not match any detections). By varying
    the score threshold s_i over all detection scores, we have the PR curve.

    What does it mean for a detection to match a ground truth label? In this assignment, we use
    the following definition: A detection matches a ground truth label if: (1) the Euclidean
    distance between their centers is at most `threshold`; and (2) no higher scoring detection
    satisfies condition (1) with respect to the same label.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A precision/recall curve.
    """
    
    # Compute the TP, FN, and DS vectors
    TP_vec, FN_vec, DS_vec = torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0)
    for frame in frames:
        detections = frame.detections
        labels = frame.labels
        
        # Distance between all combinations of centroids [N, M] shape matrix
        dist = torch.cdist(detections.centroids, labels.centroids, p=2)
        dist_lt_threshold = dist < threshold

        local_TP_vec = torch.any(dist_lt_threshold, axis=1).long()
        local_FN_vec = torch.any(dist_lt_threshold, axis=0).long()

        TP_vec = torch.concat((TP_vec, local_TP_vec), axis=0)
        FN_vec = torch.concat((FN_vec, local_FN_vec), axis=0)
        DS_vec = torch.concat((DS_vec, detections.scores), axis=0)

    # Sort the matches by the corresponding detection scores in DS_vec.
    sorting_index = torch.argsort(DS_vec).flip(dims=(0,))
    TP_vec = TP_vec[sorting_index]
    DS_vec = DS_vec[sorting_index]

    # Calculate the prescision and precision for the PRCurve
    precision = torch.zeros(TP_vec.shape)
    recall = torch.zeros(FN_vec.shape)
    fn = torch.sum(FN_vec)
    for i in range(precision.shape[0]):
        vec_subset = TP_vec[:i+1]
        tp = len(vec_subset == 1)
        fp = len(vec_subset == 0)

        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)

    return PRCurve(precision, recall)


def compute_area_under_curve(curve: PRCurve) -> float:
    """Return the area under the given curve.

    Given a `PRCurve` curve, this function computes the area under the curve as:
        AP = \sum_{i = 1}^{n} (r_i - r_{i - 1}) * p_i
    where r_i (resp. p_i) is the recall (resp. precision) of the top i detections,
    n is the total number of detections, and we set r_0 = 0.0. Intuitively, this
    is computing the integral of the step function defined by the PRCurve.

    Args:
        curve: The precision/recall curve.

    Returns:
        The area under the curve, as defined above.
    """
    ri = curve.recall
    ri_minus_one = torch.zeros(ri.shape[0])
    ri_minus_one[1:] = ri[:-1]
    
    AP = torch.sum(curve.precision * (ri - ri_minus_one))

    return AP


def compute_average_precision(
    frames: List[EvaluationFrame], threshold: float
) -> AveragePrecisionMetric:
    """Compute average precision over a batch of evaluation frames.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A dataclass consisting of a PRCurve and its average precision.
    """
    pr_curve = compute_precision_recall_curve(frames, threshold)
    auc = compute_area_under_curve(pr_curve)

    return AveragePrecisionMetric(auc, pr_curve)
