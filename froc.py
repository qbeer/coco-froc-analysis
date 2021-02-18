import argparse
import json
from pathlib import Path
from pprint import pprint
from matplotlib.pyplot import winter
import numpy as np
from numpy.core.arrayprint import printoptions


def load_json_from_file(file_path):
    with Path(file_path).open() as fp:
        data = json.load(fp)
    return data


def update_scores(json_data, score_thres):
    scores = []
    for pred in json_data:
        scores.append(pred['score'])
    scores = np.array(scores)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    preds = []
    for ind, pred in enumerate(json_data):
        pred['score'] = scores[ind]
        if pred['score'] > score_thres:
            preds.append(pred)

    return preds


def init_statistics(gt, categories):
    stats = {
        cat['id']: {
            'name': cat['name'],
            'LL': 0,
            'NL': 0,
            'n_images': set(),
            'n_lesions': 0
        }
        for cat in categories
    }

    for annotation in gt['annotations']:
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        stats[category_id]['n_images'].add(image_id)
        stats[category_id]['n_lesions'] += 1

    for cat_id in stats:
        stats[cat_id]['n_images'] = len(stats[cat_id]['n_images'])

    return stats


def build_pr_id_to_annotation_dict(pr):
    id_to_annotation = dict()
    for annotation in pr:
        id_to_annotation[annotation['image_id']] = []
    for annotation in pr:
        id_to_annotation[annotation['image_id']].append(annotation)
    return id_to_annotation


def build_gt_id_to_annotation_dict(gt):
    id_to_annotation = dict()
    for image in gt['images']:
        id_to_annotation[image['id']] = []
    for annotation in gt['annotations']:
        id_to_annotation[annotation['image_id']].append(annotation)
    return id_to_annotation


def get_iou_score(gt_box, pr_box):
    gt_x, gt_y, gt_w, gt_h = gt_box
    pr_x, pr_y, pr_w, pr_h = pr_box

    xA = max(gt_x, pr_x)
    xB = min(gt_x + gt_w, pr_x + pr_w)
    yA = max(gt_y, pr_y)
    yB = min(gt_y + gt_h, pr_y + pr_h)

    intersection = max((xB - xA), 0) * max((yB - yA), 0)
    if intersection == 0:
        return 0

    gt_area = gt_w * gt_h
    pr_area = pr_w * pr_h

    return intersection / (gt_area + pr_area - intersection)


def update_stats(gt_id_to_annotation, pr_id_to_annotation, stats,
                 args):
    for image_id in gt_id_to_annotation:
        for gt_ann in gt_id_to_annotation[image_id]:
            is_ll = False

            for pred_ann in pr_id_to_annotation.get(image_id, []):
                if gt_ann['category_id'] != pred_ann['category_id']:
                    continue

                if pred_ann['score'] < args.score_thres:
                    print(pred_ann['score'], args.score_thres)
                    continue

                if args.use_iou:
                    iou_score = get_iou_score(gt_ann['bbox'], pred_ann['bbox'])
                    if args.iou_thres < iou_score:
                        stats[gt_ann['category_id']]['LL'] += 1
                        is_ll = True
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
                        is_ll = True
                        break

                if not is_ll:
                    stats[gt_ann['category_id']]['NL'] += 1

    return stats


def run(args):
    gt = load_json_from_file(args.gt_ann)
    pr = load_json_from_file(args.pred_ann)

    pr = update_scores(pr, args.score_thres)

    categories = gt['categories']

    stats = init_statistics(gt, categories)

    gt_id_to_annotation = build_gt_id_to_annotation_dict(gt)
    pr_id_to_annotation = build_pr_id_to_annotation_dict(pr)

    stats = update_stats(gt_id_to_annotation, pr_id_to_annotation,
                         stats, args)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_ann', type=str, required=True)
    parser.add_argument('--pred_ann', type=str, required=True)
    parser.add_argument(
        '--use_iou',
        default=False,
        action="store_true",
        help="Use IoU score to decide on `proximity` rather then using center pixel inside GT box."
    )
    parser.add_argument(
        '--iou_thres',
        default=.75,
        type=float,
        required=False,
        help='If IoU score is used the default threshold is arbitrarily set to .75')
    parser.add_argument('--score_thres',
                        default=.5,
                        type=float,
                        required=False,
                        help='Prediction threshold')

    args = parser.parse_args()

    run(args)
