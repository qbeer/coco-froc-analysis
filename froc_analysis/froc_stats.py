from collections import Counter
from scipy.optimize import linear_sum_assignment
from .utils import get_iou_score
import numpy as np


def init_stats(gt: dict, categories: dict) -> dict:
    """Initializing the statistics before counting leasion
       and non-leasion localiazations.

    Arguments:
        gt {dict} -- Ground truth COCO dataset
        categories {dict} -- Dictionary of categories present in the COCO dataset

    Returns:
        stats {dict} -- Statistics to be updated, containing every information
                        necessary to evaluate a single FROC point
    """
    stats = {
        cat["id"]: {
            "name": cat["name"],
            "LL": 0,
            "NL": 0,
            "n_images": [],
            "n_lesions": 0,
        }
        for cat in categories
    }

    for annotation in gt["annotations"]:
        category_id = annotation["category_id"]
        stats[category_id]["n_lesions"] += 1

    for image in gt["images"]:
        image_id = image["id"]
        for cat_id in stats:
            stats[cat_id]["n_images"].append(image_id)

    for cat_id in stats:
        stats[cat_id]["n_images"] = len(stats[cat_id]["n_images"])

    return stats


def update_stats(
    stats: dict,
    gt_id_to_annotation: dict,
    pr_id_to_annotation: dict,
    categories: dict,
    use_iou: bool,
    iou_thres: float,
):
    """Updating statistics as going through images of the dataset.

    Arguments:
        stats {dict} -- FROC statistics
        gt_id_to_annotation {dict} -- Ground-truth image IDs to annotations.
        pr_id_to_annotation {dict} -- Prediction image IDs to annotations.
        categories {dict} -- COCO categories dictionary.
        use_iou {bool} -- Whether or not to use iou thresholding.
        iou_thres {float} -- IoU threshold when using IoU thresholding.

    Returns:
        stats {dict} -- Updated FROC statistics
    """
    for image_id in gt_id_to_annotation:
        cat2anns = {}
        for cat in categories:
            cat2anns[cat['id']] = {"gt": [], "pr": []}

        for gt_ann in gt_id_to_annotation[image_id]:
            cat2anns[gt_ann["category_id"]]['gt'].append(gt_ann)
        for pred_ann in pr_id_to_annotation.get(image_id, []):
            cat2anns[pred_ann["category_id"]]['pr'].append(pred_ann)

        for cat in categories:
            gt_anns = cat2anns[cat['id']]['gt']
            pr_anns = cat2anns[cat['id']]['pr']

            n_gt = len(gt_anns)
            n_pr = len(pr_anns)

            if n_gt == 0:
                if n_pr == 0:
                    continue
                stats[cat['id']]['FP'] += n_pr
            else:
                cost_matrix = np.ones((n_gt, n_pr)) * 1e6

                for gt_ind, gt_ann in enumerate(gt_anns):
                    for pr_ind, pr_ann in enumerate(pr_anns):
                        if use_iou:
                            iou_score = get_iou_score(gt_ann["bbox"],
                                                      pr_ann["bbox"])
                            if iou_score > iou_thres:
                                cost_matrix[gt_ind, pr_ind] = iou_score / (
                                    np.random.uniform(0, 1) / 1e6)
                        else:
                            gt_x, gt_y, gt_w, gt_h = gt_ann["bbox"]

                            pr_x, pr_y, pr_w, pr_h = pr_ann["bbox"]
                            pr_bbox_center = pr_x + pr_w / 2, pr_y + pr_h / 2

                            if (pr_bbox_center[0] >= gt_x
                                    and pr_bbox_center[0] <= gt_x + gt_w
                                    and pr_bbox_center[1] >= gt_y
                                    and pr_bbox_center[1] <= gt_y + gt_h):
                                cost_matrix[gt_ind, pr_ind] = 1.0

            row_ind, col_ind = linear_sum_assignment(
                cost_matrix)  # Hungarian-matching

            n_true_positives = len(row_ind)
            n_false_positives = max(n_pr - len(col_ind), 0)

            stats[cat['id']]['LL'] += n_true_positives
            stats[cat['id']]['NL'] += n_false_positives

    return stats
