from .froc_stats import init_stats, update_stats
from .utils import (
    load_json_from_file,
    update_scores,
    build_gt_id2annotations,
    build_pr_id2annotations,
)
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def froc_point(gt_ann, pr_ann, score_thres, use_iou, iou_thres):
    gt = load_json_from_file(gt_ann)
    pr = load_json_from_file(pr_ann)

    pr = update_scores(pr, score_thres)

    categories = gt["categories"]

    stats = init_stats(gt, categories)

    gt_id_to_annotation = build_gt_id2annotations(gt)
    pr_id_to_annotation = build_pr_id2annotations(pr)

    stats = update_stats(
        stats, gt_id_to_annotation, pr_id_to_annotation, categories, use_iou, iou_thres
    )

    return stats


def generate_froc_curve(
    gt_ann, pr_ann, use_iou=False,
    iou_thres=.5, n_sample_points=50,
    plot_title='FROC curve', plot_output_path='froc.png'
):
    lls_accuracy = {}
    nlls_per_image = {}

    for score_thres in tqdm(np.linspace(0.0, 1.0, n_sample_points, endpoint=False)):
        stats = froc_point(gt_ann, pr_ann, score_thres, use_iou, iou_thres)
        for category_id in stats:
            if lls_accuracy.get(category_id, None):
                lls_accuracy[category_id].append(
                    stats[category_id]["LL"] / stats[category_id]["n_lesions"]
                )
            else:
                lls_accuracy[category_id] = []
                lls_accuracy[category_id].append(
                    stats[category_id]["LL"] / stats[category_id]["n_lesions"]
                )

            if nlls_per_image.get(category_id, None):
                nlls_per_image[category_id].append(
                    stats[category_id]["NL"] / stats[category_id]["n_images"]
                )
            else:
                nlls_per_image[category_id] = []
                nlls_per_image[category_id].append(
                    stats[category_id]["NL"] / stats[category_id]["n_images"]
                )

    if plot_title:
        plt.figure(figsize=(12, 12))

    for category_id in lls_accuracy:
        lls = lls_accuracy[category_id]
        nlls = nlls_per_image[category_id]
        if plot_title:
            plt.semilogx(nlls, lls, "x--", label=stats[category_id]["name"])

    if plot_title:
        plt.legend(loc="lower right")

        plt.title(plot_title)
        plt.ylabel("Sensitivity")
        plt.xlabel("FP / per image")

        plt.tight_layout()

        plt.savefig(plot_output_path, dpi=50)
    else:
        return lls_accuracy, nlls_per_image
