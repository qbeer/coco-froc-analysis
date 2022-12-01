from __future__ import annotations

import json
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..utils import transform_gt_into_pr
from .froc_curve import calc_scores
from .froc_curve import colors
from .froc_curve import froc_point
from .froc_curve import generate_froc_curve


def generate_bootstrap_froc_curves(
    gt_ann,
    pr_ann,
    n_bootstrap_samples=5,
    use_iou=False,
    iou_thres=0.5,
    n_sample_points=50,
    plot_title='Bootstrap FROC',
    plot_output_path='froc_bootstrapped.png',
    test_ann=None,
):
    with open(gt_ann) as fp:
        GT_ANN = json.load(fp)

    with open(pr_ann) as fp:
        PRED_ANN = json.load(fp)

    n_images = len(GT_ANN['images'])

    fig, ax = plt.subplots(figsize=[27, 10])
    ins = ax.inset_axes([0.55, 0.05, 0.45, 0.4])
    ins.set_xlim([0.1, 5.0])
    ins.set_xticks([0.1, 1.0, 2.0, 3.0, 4.0])

    collected_frocs = {'lls': {}, 'nlls': {}}

    non_bootstrap_lls, non_bootstrap_nlls = generate_froc_curve(
        gt_ann,
        pr_ann,
        use_iou,
        iou_thres,
        n_sample_points,
        plot_title=None,
        plot_output_path=None,
    )

    for _ in tqdm(range(n_bootstrap_samples)):
        selected_images = random.choices(
            GT_ANN['images'], k=n_images,
        )  # sample with replacement
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

        lls, nlls = generate_froc_curve(
            tmp_gt_ann,
            tmp_pred_ann,
            use_iou,
            iou_thres,
            n_sample_points,
            plot_title=None,
            plot_output_path=None,
        )

        for cat_id in lls:
            if collected_frocs['lls'].get(cat_id, None) is None:
                collected_frocs['lls'] = {cat_id: []}
            if collected_frocs['nlls'].get(cat_id, None) is None:
                collected_frocs['nlls'] = {cat_id: []}

        for cat_id in lls:
            collected_frocs['lls'][cat_id].append(lls[cat_id])
            collected_frocs['nlls'][cat_id].append(nlls[cat_id])

    interpolated_frocs = {}
    max_froc_lls = {}
    min_froc_lls = {}

    min_nlls, max_nlls = np.min(non_bootstrap_nlls[cat_id]), np.max(
        non_bootstrap_nlls[cat_id],
    )

    x_range = np.logspace(
        np.log10(min_nlls + 1e-2), np.log10(max_nlls),
        n_sample_points, endpoint=True,
    )

    for cat_id in collected_frocs['lls']:
        all_lls = np.array(collected_frocs['lls'][cat_id]).reshape(
            n_bootstrap_samples, n_sample_points,
        )
        all_nlls = np.array(collected_frocs['nlls'][cat_id]).reshape(
            n_bootstrap_samples, n_sample_points,
        )

        frocs = []

        for lls, nlls in zip(all_lls, all_nlls):
            interpolated_lls = np.interp(x_range, nlls[::-1], lls[::-1])
            frocs.append(interpolated_lls)

        interpolated_frocs[cat_id] = {
            'nlls': x_range,
            'lls': np.array(frocs).reshape(
                n_bootstrap_samples,
                n_sample_points,
            ),
        }

        max_froc_lls[cat_id] = np.max(
            np.array(frocs).reshape(n_bootstrap_samples, n_sample_points),
            axis=0,
        )

        min_froc_lls[cat_id] = np.min(
            np.array(frocs).reshape(n_bootstrap_samples, n_sample_points),
            axis=0,
        )

    for cat_id in interpolated_frocs:
        ax.semilogx(
            interpolated_frocs[cat_id]['nlls'],
            np.mean(interpolated_frocs[cat_id]['lls'], axis=0),
            'b-',
            label='mean',
        )

        ins.plot(
            interpolated_frocs[cat_id]['nlls'],
            np.mean(interpolated_frocs[cat_id]['lls'], axis=0),
            'b-',
            label='mean',
        )

        ax.fill_between(
            interpolated_frocs[cat_id]['nlls'],
            min_froc_lls[cat_id],
            max_froc_lls[cat_id],
            alpha=.2,
        )

        ins.fill_between(
            interpolated_frocs[cat_id]['nlls'],
            min_froc_lls[cat_id],
            max_froc_lls[cat_id],
            alpha=.2,
        )

        for lls, nlls in zip(all_lls, all_nlls):
            ax.semilogx(nlls, lls, 'r-', alpha=.1)
            ins.plot(nlls, lls, 'r-', alpha=.1)

        if test_ann is not None:
            for t_ann, c in zip(test_ann, colors):
                t_pr = transform_gt_into_pr(t_ann, gt_ann)
                stats = froc_point(gt_ann, t_pr, .5, use_iou, iou_thres)
                _lls_accuracy, _nlls_per_image = calc_scores(stats, {}, {})
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
                    _nlls_per_image[cat_id][0],
                    _lls_accuracy[cat_id][0],
                    'D',
                    markersize=15,
                    markeredgewidth=3,
                    label=label +
                    f' (FP/image = {np.round(_nlls_per_image[cat_id][0], 2)})',
                    c=c,
                )
                ins.plot(
                    _nlls_per_image[cat_id][0],
                    _lls_accuracy[cat_id][0],
                    'D',
                    markersize=12,
                    markeredgewidth=2,
                    label=label +
                    f' (FP/image = {np.round(_nlls_per_image[cat_id][0], 2)})',
                    c=c,
                )
                ax.hlines(
                    y=_lls_accuracy[cat_id][0],
                    xmin=min_nlls,
                    xmax=max_nlls,
                    linestyles='dashed',
                    colors=c,
                )
                ins.hlines(
                    y=_lls_accuracy[cat_id][0],
                    xmin=min_nlls,
                    xmax=max_nlls,
                    linestyles='dashed',
                    colors=c,
                )
                ax.text(
                    x=min_nlls, y=_lls_accuracy[cat_id][0] - 0.02,
                    s=f' FP/image = {np.round(_nlls_per_image[cat_id][0], 2)}',
                    fontdict={'fontsize': 20, 'fontweight': 'bold'},
                )

        ax.semilogx(
            non_bootstrap_nlls[cat_id], non_bootstrap_lls[cat_id], 'r--', label='non-bootstrap',
        )
        ins.plot(
            non_bootstrap_nlls[cat_id], non_bootstrap_lls[cat_id], 'r--', label='non-bootstrap',
        )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(
        loc='center left', bbox_to_anchor=(.95, .75),
        fancybox=True, shadow=True, ncol=1, fontsize=25,
    )

    ax.set_title(plot_title, fontdict={'fontsize': 35})
    ax.set_ylabel('Sensitivity', fontdict={'fontsize': 30})
    ax.set_xlabel('FP / image', fontdict={'fontsize': 30})

    ax.tick_params(axis='both', which='major', labelsize=30)
    ins.tick_params(axis='both', which='major', labelsize=20)

    ax.set_xlim([min_nlls, max_nlls])
    ax.set_ylim(bottom=0.05, top=1.02)
    fig.savefig(fname=plot_output_path, dpi=150)

    os.remove('/tmp/tmp_bootstrap_gt.json')
    os.remove('/tmp/tmp_bootstrap_pred.json')
