from __future__ import annotations

import json
import multiprocessing as mp
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..utils import transform_gt_into_pr
from .froc_curve import calc_scores
from .froc_curve import COLORS
from .froc_curve import froc_point
from .froc_curve import generate_froc_curve


def run_bootstrap(args):
    GT_ANN, PRED_ANN, n_images, n_sample_points, use_iou, iou_thres = args
    selected_images = random.choices(
        GT_ANN['images'],
        k=n_images,
    )
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

    lls, nlls = generate_froc_curve(
        bootstrap_gt,
        predictions,
        use_iou,
        iou_thres,
        n_sample_points,
        plot_title=None,
        plot_output_path=None,
    )

    return lls, nlls


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
    bounds=None,
):
    with open(gt_ann) as fp:
        GT_ANN = json.load(fp)

    with open(pr_ann) as fp:
        PRED_ANN = json.load(fp)

    n_images = len(GT_ANN['images'])

    fig, ax = plt.subplots(figsize=[27, 18])
    ins = ax.inset_axes([0.55, 0.05, 0.45, 0.4])
    ins.set_xticks(
        [0.1, 1.0, 2.0, 3.0, 4.0],
        [
            0.1,
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        fontsize=45,
    )

    ins.set_xlim([0.1, 4.5])

    _, non_bootstrap_nlls = generate_froc_curve(
        gt_ann,
        pr_ann,
        use_iou,
        iou_thres,
        n_sample_points,
        plot_title=None,
        plot_output_path=None,
    )

    collected_frocs = {'lls': {}, 'nlls': {}}

    with mp.Pool(mp.cpu_count() // 4 + 1) as pool:
        args_list = [(
            GT_ANN, PRED_ANN, n_images, n_sample_points,
            use_iou, iou_thres,
        ) for _ in range(n_bootstrap_samples)]
        for outputs in tqdm(
            pool.imap(run_bootstrap, args_list),
            total=n_bootstrap_samples,
            desc='Evaluating bootstrap samples...',
        ):
            lls, nlls = outputs
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

    if bounds is not None:
        min_nlls, max_nlls = bounds[0], bounds[1]

    x_range = np.logspace(
        np.log10(min_nlls + 1e-8),
        np.log10(max_nlls),
        n_sample_points,
        endpoint=True,
    )

    for cat_id in collected_frocs['lls']:
        all_lls = np.array(collected_frocs['lls'][cat_id]).reshape(
            n_bootstrap_samples,
            n_sample_points,
        )
        all_nlls = np.array(collected_frocs['nlls'][cat_id]).reshape(
            n_bootstrap_samples,
            n_sample_points,
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
            alpha=0.2,
        )

        ins.fill_between(
            interpolated_frocs[cat_id]['nlls'],
            min_froc_lls[cat_id],
            max_froc_lls[cat_id],
            alpha=0.2,
        )

        if test_ann is not None:
            for t_ann, c in zip(test_ann, COLORS):
                t_ann, label = t_ann
                t_pr = transform_gt_into_pr(t_ann, gt_ann)
                stats = froc_point(gt_ann, t_pr, 0.5, use_iou, iou_thres)
                _lls_accuracy, _nlls_per_image = calc_scores(stats, {}, {})
                ax.plot(
                    _nlls_per_image[cat_id][0],
                    _lls_accuracy[cat_id][0],
                    'D',
                    markersize=15,
                    markeredgewidth=3,
                    label=label
                    + f' (FP/image = {np.round(_nlls_per_image[cat_id][0], 2)})',
                    c=c,
                )
                ins.plot(
                    _nlls_per_image[cat_id][0],
                    _lls_accuracy[cat_id][0],
                    'D',
                    markersize=12,
                    markeredgewidth=2,
                    label=label
                    + f' (FP/image = {np.round(_nlls_per_image[cat_id][0], 2)})',
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

                if bounds is not None:
                    min_rec, _ = bounds[0], bounds[1]
                    if _nlls_per_image[cat_id][0] < min_rec:
                        continue
                else:
                    ax.text(
                        x=_nlls_per_image[cat_id][0],
                        y=_lls_accuracy[cat_id][0],
                        s=f' FP/image = {np.round(_nlls_per_image[cat_id][0], 2)}',
                        fontdict={'fontsize': 40, 'fontweight': 'bold'},
                    )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(
        loc='lower left',
        bbox_to_anchor=(0.1, 0.1),
        fancybox=True,
        shadow=True,
        ncol=1,
        fontsize=45,
    )

    ax.set_title(plot_title, fontdict={'fontsize': 45})
    ax.set_ylabel('Sensitivity', fontdict={'fontsize': 40})
    ax.set_xlabel('FP / image', fontdict={'fontsize': 40})

    ax.tick_params(axis='both', which='major', labelsize=40)
    ins.tick_params(axis='both', which='major', labelsize=35)

    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([x_min, x_max])
    else:
        ax.set_ylim(bottom=0.05, top=1.02)

    ax.grid(True, which='both', axis='both', alpha=0.5, linestyle='--')

    fig.tight_layout()
    fig.savefig(fname=plot_output_path, dpi=150)
