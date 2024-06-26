from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..utils import build_gt_id2annotations
from ..utils import build_pr_id2annotations
from ..utils import COLORS
from ..utils import load_json_from_file
from ..utils import transform_gt_into_pr
from ..utils import update_scores
from .count_stats import init_stats
from .count_stats import update_stats


def count_point(gt_ann, pr_ann, score_thres, weighted):
    """
    Computes statistics based on ground truth annotations and predicted annotations.

    Parameters:
    - gt_ann (str): Path to the ground truth annotations file in JSON format.
    - pr_ann (str): Path to the predicted annotations file in JSON format.
    - score_thres (float): Score threshold for predicted annotations.
    - weighted (bool): Flag indicating whether to compute weighted count statistics.

    Returns:
    - stats (dict): Dictionary containing statistics for each category.

    This function loads ground truth and predicted annotations from JSON files,
    updates the predicted scores based on the given score threshold, initializes
    count statistics, builds dictionaries mapping annotation IDs to annotations,
    and updates the statistics based on matched annotations. It returns the computed
    count statistics.
    """
    gt = load_json_from_file(gt_ann)
    pr = load_json_from_file(pr_ann)

    pr = update_scores(pr, score_thres)

    categories = gt['categories']

    stats = init_stats(gt, categories)

    gt_id_to_annotation = build_gt_id2annotations(gt)
    pr_id_to_annotation = build_pr_id2annotations(pr)

    stats = update_stats(
        stats, gt_id_to_annotation, pr_id_to_annotation,
        categories, weighted,
    )

    return stats


def calc_scores(stats, precision, recall):
    """
    Calculates precision and recall based on count statistics.

    Parameters:
    - stats (dict): Count statistics for each category.
    - precision (dict): Dictionary to store precision per category.
    - recall (dict): Dictionary to store recall per category.

    Returns:
    - precision (dict): Updated precision dictionary.
    - recall (dict): Updated recall dictionary.

    This function calculates precision and recall for each category based on the provided
    count statistics. Precision represents the proportion of true positive counts divided
    by the sum of true positive and false positive counts. Recall represents the proportion
    of true positive counts divided by the sum of true positive and false negative counts.
    It updates the respective dictionaries and returns the updated values.
    """
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


def generate_count_curve(
    gt_ann,
    pr_ann,
    weighted=False,
    n_sample_points=50,
    plot_title='Count curve',
    plot_output_path='counts.png',
    test_ann=None,
    bounds=None,
):
    """
    Generates a count curve based on ground truth annotations and predicted annotations.

    Parameters:
    - gt_ann (str): Path to the ground truth annotations file in JSON format.
    - pr_ann (str): Path to the predicted annotations file in JSON format.
    - weighted (bool): Flag indicating whether to compute weighted count statistics.
      Default value is False.
    - n_sample_points (int): Number of sample points to generate on the count curve.
      Default value is 50.
    - plot_title (str): Title for the generated count curve plot. Default title is "Count curve".
    - plot_output_path (str): Path to save the generated count curve plot. Default path is
      "counts.png".
    - test_ann (list): List of tuples containing additional annotations for testing,
      where each tuple contains annotation data and a label. Default value is None.
    - bounds (tuple): Tuple containing the minimum and maximum bounds for the x and y
      axes of the plot, respectively. Default value is None.

    Returns:
    - None

    This function generates a count curve based on ground truth annotations (`gt_ann`)
    and predicted annotations (`pr_ann`). It computes precision and recall at different
    operating points along the curve. The generated curve is plotted with optional
    additional annotations for testing. The resulting plot is saved at the specified
    `plot_output_path`.
    """
    precision = {}
    recall = {}

    for score_thres in tqdm(
            np.linspace(0.0, 1.0, n_sample_points, endpoint=False),
    ):
        stats = count_point(gt_ann, pr_ann, score_thres, weighted)
        precision, recall = calc_scores(stats, precision, recall)

    if plot_title:
        fig, ax = plt.subplots(figsize=[27, 18])
        ins = ax.inset_axes([0.05, 0.05, 0.45, 0.4])
        ins.set_xticks(
            [.7, .75, .8, .85, .9, .95],
            [.7, .75, .8, .85, .9, .95], fontsize=30,
        )
        ins.yaxis.tick_right()
        ins.xaxis.tick_top()
        if bounds is not None:
            _, x_max, _, _ = bounds
            ins.set_xlim([.8, x_max])
        else:
            ins.set_xlim([.8, 1.0])

    for category_id in precision:
        prec = precision[category_id]
        rec = recall[category_id]
        if plot_title:
            ax.plot(
                rec,
                prec,
                'x--',
                label='AI ' + stats[category_id]['name'],
            )
            ins.plot(
                rec,
                prec,
                'x--',
                label='AI ' + stats[category_id]['name'],
            )

            if test_ann is not None:
                for t_ann, c in zip(test_ann, COLORS):
                    t_ann, label = t_ann
                    t_pr = transform_gt_into_pr(t_ann, gt_ann)
                    stats = count_point(gt_ann, t_pr, .5, weighted)
                    _precision, _recall = calc_scores(stats, {}, {})
                    if plot_title:
                        ax.plot(
                            _recall[category_id][0],
                            _precision[category_id][0],
                            'D',
                            markersize=15,
                            markeredgewidth=3,
                            label=label +
                            f' (R = {np.round(_recall[category_id][0], 3)})',
                            c=c,
                        )
                        ins.plot(
                            _recall[category_id][0],
                            _precision[category_id][0],
                            'D',
                            markersize=12,
                            markeredgewidth=2,
                            label=label +
                            f' (R = {np.round(_recall[category_id][0], 3)})',
                            c=c,
                        )
                        ax.hlines(
                            y=_precision[category_id][0],
                            xmin=np.min(rec),
                            xmax=np.max(rec),
                            linestyles='dashed',
                            colors=c,
                        )
                        ax.text(
                            x=_recall[category_id][0], y=_precision[category_id][0],
                            s=f' R = {np.round(_recall[category_id][0], 3)}',
                            fontdict={'fontsize': 20, 'fontweight': 'bold'},
                        )
                        ins.hlines(
                            y=_precision[category_id][0],
                            xmin=np.min(rec),
                            xmax=np.max(rec),
                            linestyles='dashed',
                            colors=c,
                        )

    if plot_title:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.legend(
            loc='lower left', bbox_to_anchor=(.65, .1),
            fancybox=True, shadow=True, ncol=1, fontsize=30,
        )

        ax.set_title(plot_title, fontdict={'fontsize': 35})
        ax.set_ylabel(
            'Precision', fontdict={
                'fontsize': 30,
            },
        )
        ax.set_xlabel('Recall', fontdict={'fontsize': 30})

        ax.tick_params(axis='both', which='major', labelsize=30)
        ins.tick_params(axis='both', which='major', labelsize=20)

        if bounds is not None:
            x_min, x_max, _, _ = bounds
            ax.set_xlim([x_min, x_max])
        else:
            ax.set_xlim([.7, 1.0])
            ax.set_ylim(bottom=0.05, top=1.02)
        fig.tight_layout()
        fig.savefig(plot_output_path, dpi=150)
    else:
        return precision, recall
