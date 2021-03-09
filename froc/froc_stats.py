import argparse
import json
import numpy as np
from collections import Counter

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
        cat['id']: {
            'name': cat['name'],
            'LL': 0,
            'NL': 0,
            'n_images': [],
            'n_lesions': 0
        }
        for cat in categories
    }

    for annotation in gt['annotations']:
        category_id = annotation['category_id']
        stats[category_id]['n_lesions'] += 1

    for image in gt['images']:
        image_id = image['id']
        for cat_id in stats:
            stats[cat_id]['n_images'].append(image_id)

    for cat_id in stats:
        stats[cat_id]['n_images'] = len(stats[cat_id]['n_images'])

    return stats


def update_stats(stats : dict,
                 gt_id_to_annotation : dict,
                 pr_id_to_annotation : dict,
                 categories : dict,
                 use_iou : bool,
                 iou_thres: float):
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
        n_is_ll = dict()
        for cat in categories:
            n_is_ll[cat['id']] = 0
        
        for gt_ann in gt_id_to_annotation[image_id]:
            for pred_ann in pr_id_to_annotation.get(image_id, []):
                if gt_ann['category_id'] != pred_ann['category_id']:
                    continue

                if use_iou:
                    iou_score = get_iou_score(gt_ann['bbox'], pred_ann['bbox'])
                    if iou_thres < iou_score:
                        stats[gt_ann['category_id']]['LL'] += 1
                        n_is_ll[gt_ann['category_id']] += 1
                        break
                else:
                    gt_x, gt_y, gt_w, gt_h = gt_ann['bbox']

                    pr_x, pr_y, pr_w, pr_h = pred_ann['bbox']
                    pr_bbox_center = pr_x + pr_w / 2, pr_y + pr_h / 2

                    if pr_bbox_center[0] >= gt_x and \
                            pr_bbox_center[0] <= gt_x + gt_w and \
                            pr_bbox_center[1] >= gt_y and \
                            pr_bbox_center[1] <= gt_y + gt_h:
                        stats[gt_ann['category_id']]['LL'] += 1
                        n_is_ll[gt_ann['category_id']] += 1
                        break

        cat_to_n_pred = Counter([pr_ann['category_id'] for pr_ann in pr_id_to_annotation.get(image_id, [])])

        difference = cat_to_n_pred - Counter(n_is_ll)
        for cat_id in difference:
            stats[cat_id]['NL'] += difference[cat_id]

    return stats
