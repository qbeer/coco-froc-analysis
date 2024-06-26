from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils import get_overlap


def init_stats(gt: dict, categories: dict) -> dict:
    """
    Initializes statistics for each category based on ground truth annotations.

    Parameters:
    - gt (dict): Ground truth annotations containing information about categories.
    - categories (dict): Dictionary containing category information.

    Returns:
    - stats (dict): Dictionary containing initialized statistics for each category.

    This function initializes statistics for each category based on the provided
    ground truth annotations and categories information. It creates a dictionary
    with keys as category IDs and values as dictionaries containing the category name
    along with initialized counts for true positives (TP), false positives (FP),
    false negatives (FN), and total predictions (P).
    """
    stats = {
        cat['id']: {
            'name': cat['name'],
            'P': 0,
            'TP': 0,
            'FP': 0,
            'FN': 0,
        }
        for cat in categories
    }

    return stats


def update_stats(
    stats: dict, gt_id_to_annotation: dict,
    pr_id_to_annotation: dict, categories: dict, weighted: bool,
):
    """
    Updates statistics based on ground truth and predicted annotations.

    Parameters:
    - stats (dict): Current statistics for each category.
    - gt_id_to_annotation (dict): Mapping of image IDs to ground truth annotations.
    - pr_id_to_annotation (dict): Mapping of image IDs to predicted annotations.
    - categories (dict): Dictionary containing category information.
    - weighted (bool): Flag indicating whether to use weighted matching.

    Returns:
    - stats (dict): Updated statistics for each category.

    This function updates statistics for each category based on the provided
    ground truth and predicted annotations. It creates a cost matrix between
    ground truth annotations and predicted annotations based on overlap area
    if `weighted` is True, or Euclidean distance if False. It then applies
    the Hungarian algorithm to find the optimal assignment of annotations,
    minimizing the total cost. The computed true positives (TP), false positives
    (FP), and false negatives (FN) are updated accordingly in the `stats`
    dictionary and returned.
    """

    for image_id in gt_id_to_annotation:
        cat2anns: dict[int, dict[str, list]] = {}
        for cat in categories:
            cat2anns[cat['id']] = {'gt': [], 'pr': []}

        for gt_ann in gt_id_to_annotation[image_id]:
            cat2anns[gt_ann['category_id']]['gt'].append(gt_ann)
        for pred_ann in pr_id_to_annotation.get(image_id, []):
            cat2anns[pred_ann['category_id']]['pr'].append(pred_ann)

        for cat in categories:
            gt_anns = cat2anns[cat['id']]['gt']
            pr_anns = cat2anns[cat['id']]['pr']

            n_gt = len(gt_anns)
            n_pr = len(pr_anns)

            if n_gt == 0:
                if n_pr == 0:
                    continue
                else:
                    stats[cat['id']]['FP'] += n_pr
            else:
                cost_matrix = np.ones((n_gt, n_pr)) * np.finfo(np.float64).max

                for gt_ind, gt_ann in enumerate(gt_anns):
                    for pr_ind, pr_ann in enumerate(pr_anns):
                        if weighted:
                            overlap = get_overlap(
                                gt_ann['bbox'],
                                pr_ann['bbox'],
                            )
                            if overlap > 0.:
                                weight = 1. / (
                                    overlap +
                                    np.random.uniform(0, 1) / 1e-6
                                )
                                cost_matrix[gt_ind, pr_ind] = weight
                        else:
                            gt_x, gt_y, gt_w, gt_h = gt_ann['bbox']

                            pr_x, pr_y, pr_w, pr_h = pr_ann['bbox']
                            pr_bbox_center = pr_x + pr_w / 2, pr_y + pr_h / 2

                            if (
                                pr_bbox_center[0] >= gt_x
                                and pr_bbox_center[0] <= gt_x + gt_w
                                and pr_bbox_center[1] >= gt_y
                                and pr_bbox_center[1] <= gt_y + gt_h
                            ):
                                cost_matrix[
                                    gt_ind,
                                    pr_ind,
                                ] = 1.0  # connected, not weighted

                row_ind, col_ind = linear_sum_assignment(
                    cost_matrix,
                )  # Hungarian-matching

                n_true_positives = len(row_ind)
                n_false_positives = max(n_pr - len(col_ind), 0)
                n_false_negatives = max(n_gt - len(row_ind), 0)

                stats[cat['id']]['P'] += n_gt
                stats[cat['id']]['TP'] += n_true_positives
                stats[cat['id']]['FP'] += n_false_positives
                stats[cat['id']]['FN'] += n_false_negatives

    return stats
