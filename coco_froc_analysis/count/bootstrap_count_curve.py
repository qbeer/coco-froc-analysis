from __future__ import annotations

import json
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..utils import COLORS
from ..utils import transform_gt_into_pr
from .count_curve import calc_scores
from .count_curve import count_point
from .count_curve import generate_count_curve


def generate_bootstrap_count_curves(
    gt_ann,
    pr_ann,
    weighted=False,
    n_sample_points=50,
    n_bootstrap_samples=5,
    plot_title='Count curve',
    plot_output_path='counts.png',
    test_ann=None,
    bounds=None,
):
    """
    This function generates bootstrapped count curves, which are statistical estimates of precision-recall curves
    obtained by resampling the data with replacement. It takes ground truth and predicted annotations as input and
    performs bootstrapping to generate multiple count curves. The mean count curve and confidence intervals are
    plotted based on the bootstrapped samples.

    Args:
        gt_ann (str): Path to the ground truth annotation JSON file.
        pr_ann (str): Path to the predicted annotation JSON file.
        weighted (bool, optional): Whether to use weighted counts. Default value is False.
        n_sample_points (int, optional): Number of sample points for the count curves. Default value is 50.
        n_bootstrap_samples (int, optional): Number of bootstrap samples. Default value is 5.
        plot_title (str, optional): Title for the plot. Default title is 'Count curve'.
        plot_output_path (str, optional): Output path for the plot. Default path is 'counts.png'.
        test_ann (list, optional): List of test annotations for additional points on the plot.
                                   Each entry is a tuple of annotations and label. Default value is None.
        bounds (tuple, optional): Bounds for the plot in the format (x_min, x_max, y_min, y_max). Default value is None.

    Returns:
        None
    """
    with open(gt_ann) as fp:
        GT_ANN = json.load(fp)

    with open(pr_ann) as fp:
        PRED_ANN = json.load(fp)

    n_images = len(GT_ANN['images'])

    fig, ax = plt.subplots(figsize=[27, 18])
    ins = ax.inset_axes([0.05, 0.05, 0.45, 0.4])
    ins.set_xticks([0.85, 0.9, 0.95], [0.85, 0.9, 0.95], fontsize=40)
    ins.yaxis.tick_right()
    ins.xaxis.tick_top()

    if bounds is not None:
        _, x_max, _, _ = bounds
        ins.set_xlim([0.8, x_max])
    else:
        ins.set_xlim([0.8, 1.0])

    collected_rocs = {'precision': {}, 'recall': {}}

    _, non_bootstrap_rec = generate_count_curve(
        gt_ann,
        pr_ann,
        weighted=weighted,
        n_sample_points=n_sample_points,
        plot_title=None,
        plot_output_path=None,
    )

    for _ in tqdm(range(n_bootstrap_samples)):
        selected_images = random.choices(GT_ANN['images'], k=n_images)
        bootstrap_gt = deepcopy(GT_ANN)

        del bootstrap_gt['images']

        bootstrap_gt['images'] = selected_images

        gt_annotations = bootstrap_gt['annotations']

        del bootstrap_gt['annotations']

        bootstrap_gt['annotations'] = []
        for _gt_ann_ in gt_annotations:
            img_id = _gt_ann_['image_id']
            for ind, selected_image in enumerate(selected_images):
                if selected_image['id'] == img_id:
                    new_gt_ann = deepcopy(_gt_ann_)
                    new_gt_ann['image_id'] = ind
                    bootstrap_gt['annotations'].append(new_gt_ann)

        predictions = []

        for ind, img in enumerate(selected_images):
            for pr in PRED_ANN:
                if pr['image_id'] == img['id']:
                    new_pr = deepcopy(pr)
                    new_pr['image_id'] = ind
                    predictions.append(new_pr)

        re_indexed_images = []
        for ind in range(len(selected_images)):
            image = deepcopy(selected_images[ind])
            image['id'] = ind
            re_indexed_images.append(image)

        bootstrap_gt['images'] = re_indexed_images

        with open('/tmp/tmp_bootstrap_gt.json', 'w') as fp:
            json.dump(bootstrap_gt, fp)

        with open('/tmp/tmp_bootstrap_pred.json', 'w') as fp:
            json.dump(predictions, fp)

        tmp_gt_ann = '/tmp/tmp_bootstrap_gt.json'
        tmp_pred_ann = '/tmp/tmp_bootstrap_pred.json'

        precision, recall = generate_count_curve(
            tmp_gt_ann,
            tmp_pred_ann,
            weighted=weighted,
            n_sample_points=n_sample_points,
            plot_title=None,
            plot_output_path=None,
        )

        for cat_id in precision:
            if collected_rocs['precision'].get(cat_id, None) is None:
                collected_rocs['precision'] = {cat_id: []}
            if collected_rocs['recall'].get(cat_id, None) is None:
                collected_rocs['recall'] = {cat_id: []}

        for cat_id in precision:
            collected_rocs['precision'][cat_id].append(precision[cat_id])
            collected_rocs['recall'][cat_id].append(recall[cat_id])

    interpolated_rocs = {}
    max_roc_prec = {}
    min_roc_prec = {}

    for cat_id in collected_rocs['precision']:
        all_prec = np.array(collected_rocs['precision'][cat_id]).reshape(
            n_bootstrap_samples,
            n_sample_points,
        )
        all_rec = np.array(collected_rocs['recall'][cat_id]).reshape(
            n_bootstrap_samples,
            n_sample_points,
        )

        min_rec, max_rec = np.min(non_bootstrap_rec[cat_id]), np.max(
            non_bootstrap_rec[cat_id],
        )

        if bounds is not None:
            min_rec, max_rec = bounds[0], bounds[1]

        x_range = np.linspace(min_rec, max_rec, n_sample_points, endpoint=True)

        rocs = []

        for prec, rec in zip(all_prec, all_rec):
            interpolated_prec = np.interp(x_range, rec[::-1], prec[::-1])
            rocs.append(interpolated_prec)

        interpolated_rocs[cat_id] = {
            'rec': x_range,
            'prec': np.array(rocs).reshape(
                n_bootstrap_samples,
                n_sample_points,
            ),
        }

        max_roc_prec[cat_id] = np.max(
            np.array(rocs).reshape(n_bootstrap_samples, n_sample_points),
            axis=0,
        )

        min_roc_prec[cat_id] = np.min(
            np.array(rocs).reshape(n_bootstrap_samples, n_sample_points),
            axis=0,
        )

    mean_roc_curve = {}

    for cat_id in interpolated_rocs:
        mean_roc_curve[cat_id] = np.stack(
            (
                interpolated_rocs[cat_id]['rec'],
                np.mean(interpolated_rocs[cat_id]['prec'], axis=0),
            ),
            axis=-1,
        )

        ax.plot(
            mean_roc_curve[cat_id][:, 0],
            mean_roc_curve[cat_id][:, 1],
            'b-',
            label='mean',
        )

        ins.plot(
            mean_roc_curve[cat_id][:, 0],
            mean_roc_curve[cat_id][:, 1],
            'b-',
            label='mean',
        )

        ax.fill_between(
            interpolated_rocs[cat_id]['rec'],
            min_roc_prec[cat_id],
            max_roc_prec[cat_id],
            alpha=0.2,
        )

        ins.fill_between(
            interpolated_rocs[cat_id]['rec'],
            min_roc_prec[cat_id],
            max_roc_prec[cat_id],
            alpha=0.2,
        )

        if test_ann is not None:
            for t_ann, c in zip(test_ann, COLORS):
                t_ann, label = t_ann
                t_pr = transform_gt_into_pr(t_ann, gt_ann)
                stats = count_point(gt_ann, t_pr, 0.5, weighted)
                _prec_accuracy, _rec_per_image = calc_scores(stats, {}, {})
                ax.plot(
                    _rec_per_image[cat_id][0],
                    _prec_accuracy[cat_id][0],
                    'D',
                    markersize=15,
                    markeredgewidth=3,
                    label=label +
                    f' (R = {np.round(_rec_per_image[cat_id][0], 3)})',
                    c=c,
                )
                ins.plot(
                    _rec_per_image[cat_id][0],
                    _prec_accuracy[cat_id][0],
                    'D',
                    markersize=15,
                    markeredgewidth=3,
                    label=label +
                    f' (R = {np.round(_rec_per_image[cat_id][0], 3)})',
                    c=c,
                )

                if bounds is not None:
                    min_rec, max_rec = bounds[0], bounds[1]
                    if _rec_per_image[cat_id][0] < min_rec:
                        continue
                else:
                    ax.text(
                        x=_rec_per_image[cat_id][0],
                        y=_prec_accuracy[cat_id][0],
                        s=f' R = {np.round(_rec_per_image[cat_id][0], 3)}',
                        fontdict={'fontsize': 35, 'fontweight': 'bold'},
                    )

                ax.hlines(
                    y=_prec_accuracy[cat_id][0],
                    xmin=min_rec,
                    xmax=max_rec,
                    linestyles='dashed',
                    colors=c,
                )

                ins.hlines(
                    y=_prec_accuracy[cat_id][0],
                    xmin=min_rec,
                    xmax=max_rec,
                    linestyles='dashed',
                    colors=c,
                )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(
        loc='lower left',
        bbox_to_anchor=(0.65, 0.1),
        fancybox=True,
        shadow=True,
        ncol=1,
        fontsize=40,
    )

    ax.set_title(plot_title, fontdict={'fontsize': 45})
    ax.set_ylabel(
        'Precision',
        fontdict={
            'fontsize': 45,
        },
    )
    ax.set_xlabel('Recall', fontdict={'fontsize': 45})

    ax.tick_params(axis='both', which='major', labelsize=40)
    ins.tick_params(axis='both', which='major', labelsize=40)

    if bounds is not None:
        x_min, x_max, _, _ = bounds
        ax.set_xlim([x_min, x_max])
    else:
        ax.set_xlim([0.7, 1.0])
        ax.set_ylim(bottom=0.05, top=1.02)

    ax.grid(True, which='both', axis='both', alpha=0.5, linestyle='--')
    fig.tight_layout()
    fig.savefig(plot_output_path, dpi=150)

    os.remove('/tmp/tmp_bootstrap_gt.json')
    os.remove('/tmp/tmp_bootstrap_pred.json')
