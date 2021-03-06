import json
from pathlib import Path


def transform_gt_into_pr(gt: str, true_gt: str) -> str:
    gt = load_json_from_file(gt)
    true_gt = load_json_from_file(true_gt)
    id2trueId = {}
    for img in true_gt['images']:
        for gt_img in gt['images']:
            if img['path'].split('/')[-1] == gt_img['path'].split('/')[-1]:
                id2trueId[gt_img['id']] = img['id']
    annotations = gt['annotations']
    scored_annotations = []
    for anno in annotations:
        anno['score'] = 1.0
        anno['category_id'] = 1
        anno['image_id'] = id2trueId[anno['image_id']]
        scored_annotations.append(anno)
    with open('/tmp/pr2gt.json', 'w') as fp:
        json.dump(scored_annotations, fp)
    return '/tmp/pr2gt.json'


def load_json_from_file(file_path):
    with Path(file_path).open() as fp:
        data = json.load(fp)
    return data


def update_scores(json_data: list, score_thres: float) -> list:
    preds = []
    for ind, pred in enumerate(json_data):
        if pred["score"] > score_thres:
            preds.append(pred)
    return preds


def get_overlap(gt_box: list, pr_box: list) -> float:
    """Intersection score between GT and prediction boxes.

    Arguments:
        gt_box {list} -- [x, y, w, h] of ground-truth lesion
        pr_box {list} -- [x, y, w, h] of prediction bounding box

    Returns:
        intersection {float}
    """
    gt_x, gt_y, gt_w, gt_h = gt_box
    pr_x, pr_y, pr_w, pr_h = pr_box

    xA = max(gt_x, pr_x)
    xB = min(gt_x + gt_w, pr_x + pr_w)
    yA = max(gt_y, pr_y)
    yB = min(gt_y + gt_h, pr_y + pr_h)

    return float(max((xB - xA), 0) * max((yB - yA), 0))


def get_iou_score(gt_box: list, pr_box: list) -> float:
    """IoU score between GT and prediction boxes.

    Arguments:
        gt_box {list} -- [x, y, w, h] of ground-truth lesion
        pr_box {list} -- [x, y, w, h] of prediction bounding box

    Returns:
        score {float} -- intersection over union score of the two boxes
    """
    *_, gt_w, gt_h = gt_box
    *_, pr_w, pr_h = pr_box

    intersection = get_overlap(gt_box, pr_box)

    gt_area = gt_w * gt_h
    pr_area = pr_w * pr_h

    return intersection / (gt_area + pr_area - intersection)


def build_pr_id2annotations(pr: list) -> dict:
    """Build image to annotation dictionary based on list of predictions.

    Arguments:
        pr {list} -- List of predictions from Detectron2 (coco-instance-results.json)

    Returns:
        id_to_annotation {dict} -- Image IDs to predicted annotations.
    """
    id_to_annotation = dict()
    for annotation in pr:
        id_to_annotation[annotation["image_id"]] = []
    for annotation in pr:
        id_to_annotation[annotation["image_id"]].append(annotation)
    return id_to_annotation


def build_gt_id2annotations(gt: dict) -> dict:
    """Build image to annotation dictionary based on the ground truth dataset.

    Arguments:
        gt {dict} -- COCO dataset of ground truth annotations.

    Returns:
        id_to_annotation {dict} -- Image IDs to ground-truth annotations.
    """
    id_to_annotation = dict()
    for image in gt["images"]:
        id_to_annotation[image["id"]] = []
    for annotation in gt["annotations"]:
        id_to_annotation[annotation["image_id"]].append(annotation)
    return id_to_annotation
