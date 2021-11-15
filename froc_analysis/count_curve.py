from .count_stats import init_stats, update_stats
from .utils import (load_json_from_file, build_gt_id2annotations,
                    update_scores, build_pr_id2annotations,
                    transform_gt_into_pr)
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def count_point(gt_ann, pr_ann, score_thres, weighted):
    gt = load_json_from_file(gt_ann)
    pr = load_json_from_file(pr_ann)

    pr = update_scores(pr, score_thres)

    categories = gt["categories"]

    stats = init_stats(gt, categories)

    gt_id_to_annotation = build_gt_id2annotations(gt)
    pr_id_to_annotation = build_pr_id2annotations(pr)

    stats = update_stats(stats, gt_id_to_annotation, pr_id_to_annotation,
                         categories, weighted)

    return stats


def calc_scores(stats, precision, recall):
    for category_id in stats:
        tp = stats[category_id]['TP']
        fp = stats[category_id]['FP']
        fn = stats[category_id]['FN']

        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)

        if precision.get(category_id, None) is None:
            precision[category_id] = []
        precision[category_id].append(prec)

        if recall.get(category_id, None) is None:
            recall[category_id] = []
        recall[category_id].append(rec)

    return precision, recall


def generate_count_curve(gt_ann,
                         pr_ann,
                         weighted=False,
                         n_sample_points=50,
                         plot_title="Count curve",
                         plot_output_path="counts.png",
                         test_ann=None):
    precision = {}
    recall = {}

    for score_thres in tqdm(
            np.linspace(0.0, 1.0, n_sample_points, endpoint=False)):
        stats = count_point(gt_ann, pr_ann, score_thres, weighted)
        precision, recall = calc_scores(stats, precision, recall)

    if plot_title:
        plt.figure(figsize=(12, 12))

    for category_id in precision:
        prec = precision[category_id]
        rec = recall[category_id]
        if plot_title:
            plt.plot(prec,
                     rec,
                     "x--",
                     label='AI ' + stats[category_id]["name"])

            if test_ann is not None:
                for t_ann in test_ann:
                    t_pr = transform_gt_into_pr(t_ann, gt_ann)
                    stats = count_point(gt_ann, t_pr, .5, weighted)
                    _precision, _recall = calc_scores(stats, {}, {})
                    label = t_ann.split('/')[-1].replace('.json', '')
                    plt.plot(_precision[category_id][0],
                             _recall[category_id][0],
                             '+',
                             markersize=12,
                             label=label)

    if plot_title:
        plt.legend(loc="lower right")

        plt.title(plot_title)
        plt.ylabel("Precision")
        plt.xlabel("Recall")

        plt.tight_layout()

        plt.xlim(0.01, 1.01)
        plt.ylim(0.01, 1.01)

        plt.savefig(plot_output_path, dpi=50)
    else:
        return precision, recall
