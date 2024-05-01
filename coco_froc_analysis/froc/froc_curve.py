from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..utils import build_gt_id2annotations
from ..utils import build_pr_id2annotations
from ..utils import COLORS
from ..utils import transform_gt_into_pr
from ..utils import update_scores
from .froc_stats import init_stats
from .froc_stats import update_stats


def froc_point(gt, pr, score_thres, use_iou, iou_thres):
    """
    Calculate statistics for a single point on the FROC curve based on ground truth and predicted annotations.

    Parameters:
    - gt (str): Path to the ground truth annotations file in JSON format.
    - pr (str): Path to the predicted annotations file in JSON format.
    - score_thres (float): Score threshold for predictions.
    - use_iou (bool): Flag indicating whether to use Intersection over Union (IoU) for matching annotations.
    - iou_thres (float): IoU threshold for matching annotations if `use_iou` is True.

    Returns:
    - dict: Statistics including true positives (TP), false positives (FP), and false negatives (FN) for each category.

    This function calculates the statistics for a single point on the FROC curve based on ground truth and predicted
    annotations. It loads the annotations from JSON files, updates the scores based on the score threshold, initializes
    FROC statistics, and then updates the statistics by comparing ground truth and predicted annotations.
    """

    if type(pr) == str:
        with open(pr) as fp:
            pr = json.load(fp)

    if type(gt) == str:
        with open(gt) as fp:
            gt = json.load(fp)

    pr = update_scores(pr, score_thres)

    categories = gt['categories']

    stats = init_stats(gt, categories)

    gt_id_to_annotation = build_gt_id2annotations(gt)
    pr_id_to_annotation = build_pr_id2annotations(pr)

    stats = update_stats(
        stats,
        gt_id_to_annotation,
        pr_id_to_annotation,
        categories,
        use_iou,
        iou_thres,
    )

    return stats


def calc_scores(stats, lls_accuracy, nlls_per_image):
    """
    Calculates Sensitivity and False Positive Rate (FPR) based on FROC statistics.

    Parameters:
    - stats (dict): FROC statistics for each category.
    - lls_accuracy (dict): Dictionary to store Sensitivity per category.
    - nlls_per_image (dict): Dictionary to store FPR per image per category.

    Returns:
    - lls_accuracy (dict): Updated Sensitivity dictionary.
    - nlls_per_image (dict): Updated FPR per image dictionary.

    This function calculates Sensitivity (True Positive Rate) and False Positive Rate (FPR)
    for each category based on the provided FROC statistics. Sensitivity represents the
    proportion of correctly detected lesions (True Positives) divided by the total number
    of lesions present in the ground truth data. FPR represents the proportion of falsely
    detected lesions (False Positives) divided by the total number of images. The function
    updates the respective dictionaries and returns the updated values.
    """
    for category_id in stats:
        if lls_accuracy.get(category_id, None):
            lls_accuracy[category_id].append(
                stats[category_id]['LL'] / stats[category_id]['n_lesions'],
            )
        else:
            lls_accuracy[category_id] = []
            lls_accuracy[category_id].append(
                stats[category_id]['LL'] / stats[category_id]['n_lesions'],
            )

        if nlls_per_image.get(category_id, None):
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] / stats[category_id]['n_images'],
            )
        else:
            nlls_per_image[category_id] = []
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] / stats[category_id]['n_images'],
            )

    return lls_accuracy, nlls_per_image


def generate_froc_curve(
    gt_ann,
    pr_ann,
    use_iou=False,
    iou_thres=0.5,
    n_sample_points=50,
    plot_title='FROC curve',
    plot_output_path='froc.png',
    test_ann=None,
    bounds=None,
):
    """
    Generates a Free-Response Receiver Operating Characteristic (FROC) curve based on
    ground truth annotations and predicted annotations.

    Parameters:
    - gt_ann (str): Path to the ground truth annotations file in JSON format.
    - pr_ann (str): Path to the predicted annotations file in JSON format.
    - use_iou (bool): Flag indicating whether to use Intersection over Union (IoU) for
      matching annotations. Default value is False.
    - iou_thres (float): IoU threshold for matching annotations if `use_iou` is True.
      Default value is 0.5.
    - n_sample_points (int): Number of sample points to generate on the FROC curve.
      Default value is 50.
    - plot_title (str): Title for the generated FROC curve plot. Default title is "FROC curve".
    - plot_output_path (str): Path to save the generated FROC curve plot. Default path is
      "froc.png".
    - test_ann (list): List of tuples containing additional annotations for testing,
      where each tuple contains annotation data and a label. Default value is None.
    - bounds (tuple): Tuple containing the minimum and maximum bounds for the x and y
      axes of the plot, respectively. Defaults value is None.

    Returns:
    - None

    This function generates a Free-Response Receiver Operating Characteristic (FROC) curve
    based on ground truth annotations (`gt_ann`) and predicted annotations (`pr_ann`). It
    computes the sensitivity and false positive rate (FPR) at different operating points
    along the curve. The generated curve is plotted with optional additional annotations
    for testing. The resulting plot is saved at the specified `plot_output_path`.
    """

    lls_accuracy = {}
    nlls_per_image = {}

    fig, ax = plt.subplots(figsize=[27, 18])

    if plot_title:
        ins = ax.inset_axes([0.55, 0.05, 0.45, 0.4])
        ins.set_xticks(
            [0.1, 1.0, 2.0, 3.0, 4.0],
            [0.1, 1.0, 2.0, 3.0, 4.0],
            fontsize=30,
        )

        if bounds is not None:
            _, x_max, _, y_max = bounds
            ins.set_xlim([0.1, x_max])
        else:
            ins.set_xlim([0.1, 4.5])

    for score_thres in tqdm(
        np.linspace(0.0, 1.0, n_sample_points, endpoint=False),
    ):
        stats = froc_point(gt_ann, pr_ann, score_thres, use_iou, iou_thres)
        lls_accuracy, nlls_per_image = calc_scores(
            stats,
            lls_accuracy,
            nlls_per_image,
        )

    for category_id in lls_accuracy:
        lls = lls_accuracy[category_id]
        nlls = nlls_per_image[category_id]

        if plot_title:
            plot_func = ax.semilogx if np.all(np.array(nlls) > 0) else ax.plot
            plot_func(
                nlls,
                lls,
                'D--',
                label='AI ' + stats[category_id]['name'],
                linewidth=4,
                markersize=25,
            )
            if ins:
                ins.plot(
                    nlls,
                    lls,
                    'D--',
                    label='AI ' + stats[category_id]['name'],
                    linewidth=4,
                    markersize=25,
                )

            if test_ann is not None:
                for t_ann, c in zip(test_ann, COLORS):
                    t_ann, label = t_ann
                    t_pr = transform_gt_into_pr(t_ann, gt_ann)
                    stats = froc_point(gt_ann, t_pr, 0.5, use_iou, iou_thres)
                    _lls_accuracy, _nlls_per_image = calc_scores(stats, {}, {})
                    if plot_title:
                        ax.plot(
                            _nlls_per_image[category_id][0],
                            _lls_accuracy[category_id][0],
                            'D',
                            markersize=15,
                            markeredgewidth=3,
                            label=label
                            + f' (FP/image = {np.round(_nlls_per_image[category_id][0], 2)})',
                            c=c,
                        )
                        if ins:
                            ins.plot(
                                _nlls_per_image[category_id][0],
                                _lls_accuracy[category_id][0],
                                'D',
                                markersize=12,
                                markeredgewidth=2,
                                label=label
                                + f' (FP/image = {np.round(_nlls_per_image[category_id][0], 2)})',
                                c=c,
                            )

                        ax.hlines(
                            y=_lls_accuracy[category_id][0],
                            xmin=np.min(nlls),
                            xmax=np.max(nlls),
                            linestyles='dashed',
                            colors=c,
                        )
                        if ins:
                            ins.hlines(
                                y=_lls_accuracy[category_id][0],
                                xmin=np.min(nlls),
                                xmax=np.max(nlls),
                                linestyles='dashed',
                                colors=c,
                            )
                        ax.text(
                            x=_nlls_per_image[category_id][0],
                            y=_lls_accuracy[category_id][0],
                            s=f' FP/image = {np.round(_nlls_per_image[category_id][0], 2)}',
                            fontdict={'fontsize': 20, 'fontweight': 'bold'},
                        )

    if plot_title:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.legend(
            loc='lower left',
            bbox_to_anchor=(0.1, 0.1),
            fancybox=True,
            shadow=True,
            ncol=1,
            fontsize=30,
        )

        ax.set_title(plot_title, fontdict={'fontsize': 35})
        ax.set_ylabel('Sensitivity', fontdict={'fontsize': 30})
        ax.set_xlabel('FP / image', fontdict={'fontsize': 30})

        ax.tick_params(axis='both', which='major', labelsize=30)
        if ins:
            ins.tick_params(axis='both', which='major', labelsize=20)

        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
            ax.set_ylim([y_min, y_max])
            ax.set_xlim([x_min, x_max])
        else:
            ax.set_ylim(bottom=-0.05, top=1.2)
        fig.tight_layout()
        fig.savefig(fname=plot_output_path, dpi=150)

    else:
        return lls_accuracy, nlls_per_image
