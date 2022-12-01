from __future__ import annotations

import json
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..utils import colors
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
):
    with open(gt_ann) as fp:
        GT_ANN = json.load(fp)

    with open(pr_ann) as fp:
        PRED_ANN = json.load(fp)

    n_images = len(GT_ANN['images'])

    fig, ax = plt.subplots(figsize=[20, 9])
    ins = ax.inset_axes([0.05, 0.05, 0.45, 0.4])
    ins.set_xlim([0.8, 1.0])
    ins.set_xticks([.85, .9, .95, 1.], fontsize=30)
    ins.set_ylim(bottom=0.2)

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
            n_bootstrap_samples, n_sample_points,
        )
        all_rec = np.array(collected_rocs['recall'][cat_id]).reshape(
            n_bootstrap_samples, n_sample_points,
        )

        min_rec, max_rec = np.min(non_bootstrap_rec[cat_id]), np.max(
            non_bootstrap_rec[cat_id],
        )

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
            alpha=.2,
        )

        ins.fill_between(
            interpolated_rocs[cat_id]['rec'],
            min_roc_prec[cat_id],
            max_roc_prec[cat_id],
            alpha=.2,
        )

        for rec, prec in zip(all_rec, all_prec):
            ax.plot(rec, prec, 'r-', alpha=.1)
            ins.plot(rec, prec, 'r-', alpha=.1)

        if test_ann is not None:
            for t_ann, c in zip(test_ann, colors):
                t_pr = transform_gt_into_pr(t_ann, gt_ann)
                stats = count_point(gt_ann, t_pr, .5, weighted)
                _prec_accuracy, _rec_per_image = calc_scores(stats, {}, {})
                label = t_ann.split('/')[-1].replace('.json', '')
                if 'bobe' in label:
                    label = 'bobe'
                elif 'istvan' in label:
                    label = 'istvan'
                elif 'tea' in label:
                    label = 'tea'
                elif 'eszter' in label:
                    label = 'eszter'
                elif 'maryam' in label:
                    label = 'maryam'
                elif 'lea' in label:
                    label = 'lea'
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
                ax.hlines(
                    y=_prec_accuracy[cat_id][0],
                    xmin=np.min(all_rec),
                    xmax=np.max(all_rec),
                    linestyles='dashed',
                    colors=c,
                )
                ax.text(
                    x=min_rec, y=_prec_accuracy[cat_id][0] - 0.035,
                    s=f' R = {np.round(_rec_per_image[cat_id][0], 3)}',
                    fontdict={'fontsize': 20, 'fontweight': 'bold'},
                )
                ins.hlines(
                    y=_prec_accuracy[cat_id][0],
                    xmin=np.min(all_rec),
                    xmax=np.max(all_rec),
                    linestyles='dashed',
                    colors=c,
                )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(
        loc='center left', bbox_to_anchor=(.95, .75),
        fancybox=True, shadow=True, ncol=1, fontsize=25,
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

    ax.set_ylim(bottom=0.05, top=1.02)
    ax.set_xlim(min_rec, 1.0)

    fig.savefig(plot_output_path, dpi=150)

    os.remove('/tmp/tmp_bootstrap_gt.json')
    os.remove('/tmp/tmp_bootstrap_pred.json')
