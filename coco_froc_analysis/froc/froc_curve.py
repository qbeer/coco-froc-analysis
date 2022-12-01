from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..utils import build_gt_id2annotations
from ..utils import build_pr_id2annotations
from ..utils import colors
from ..utils import load_json_from_file
from ..utils import transform_gt_into_pr
from ..utils import update_scores
from .froc_stats import init_stats
from .froc_stats import update_stats


def froc_point(gt_ann, pr_ann, score_thres, use_iou, iou_thres):
    gt = load_json_from_file(gt_ann)
    pr = load_json_from_file(pr_ann)

    pr = update_scores(pr, score_thres)

    categories = gt['categories']

    stats = init_stats(gt, categories)

    gt_id_to_annotation = build_gt_id2annotations(gt)
    pr_id_to_annotation = build_pr_id2annotations(pr)

    stats = update_stats(
        stats, gt_id_to_annotation, pr_id_to_annotation,
        categories, use_iou, iou_thres,
    )

    return stats


def calc_scores(stats, lls_accuracy, nlls_per_image):
    for category_id in stats:
        if lls_accuracy.get(category_id, None):
            lls_accuracy[category_id].append(
                stats[category_id]['LL'] /
                stats[category_id]['n_lesions'],
            )
        else:
            lls_accuracy[category_id] = []
            lls_accuracy[category_id].append(
                stats[category_id]['LL'] /
                stats[category_id]['n_lesions'],
            )

        if nlls_per_image.get(category_id, None):
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] /
                stats[category_id]['n_images'],
            )
        else:
            nlls_per_image[category_id] = []
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] /
                stats[category_id]['n_images'],
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
):

    lls_accuracy = {}
    nlls_per_image = {}

    for score_thres in tqdm(
            np.linspace(0.0, 1.0, n_sample_points, endpoint=False),
    ):
        stats = froc_point(gt_ann, pr_ann, score_thres, use_iou, iou_thres)
        lls_accuracy, nlls_per_image = calc_scores(
            stats, lls_accuracy,
            nlls_per_image,
        )

    if plot_title:
        fig, ax = plt.subplots(figsize=[27, 10])
        ins = ax.inset_axes([0.55, 0.05, 0.45, 0.4])
        ins.set_xlim([0.1, 5.0])
        ins.set_xticks([0.1, 1.0, 2.0, 3.0, 4.0], fontsize=30)

    for category_id in lls_accuracy:
        lls = lls_accuracy[category_id]
        nlls = nlls_per_image[category_id]
        if plot_title:
            ax.semilogx(
                nlls,
                lls,
                'x--',
                label='AI ' + stats[category_id]['name'],
            )
            ins.plot(
                nlls,
                lls,
                'x--',
                label='AI ' + stats[category_id]['name'],
            )

            ax.set_xlim(np.min(nlls), np.max(nlls))

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
                    if plot_title:
                        ax.plot(
                            _nlls_per_image[category_id][0],
                            _lls_accuracy[category_id][0],
                            'D',
                            markersize=15,
                            markeredgewidth=3,
                            label=label +
                            f' (FP/image = {np.round(_nlls_per_image[category_id][0], 2)})',
                            c=c,
                        )
                        ins.plot(
                            _nlls_per_image[category_id][0],
                            _lls_accuracy[category_id][0],
                            'D',
                            markersize=12,
                            markeredgewidth=2,
                            label=label +
                            f' (FP/image = {np.round(_nlls_per_image[category_id][0], 2)})',
                            c=c,
                        )
                        ax.hlines(
                            y=_lls_accuracy[category_id][0],
                            xmin=np.min(nlls),
                            xmax=np.max(nlls),
                            linestyles='dashed',
                            colors=c,
                        )
                        ins.hlines(
                            y=_lls_accuracy[category_id][0],
                            xmin=np.min(nlls),
                            xmax=np.max(nlls),
                            linestyles='dashed',
                            colors=c,
                        )
                        ax.text(
                            x=np.min(nlls), y=_lls_accuracy[category_id][0] - 0.02,
                            s=f' FP/image = {np.round(_nlls_per_image[category_id][0], 2)}',
                            fontdict={'fontsize': 20, 'fontweight': 'bold'},
                        )

    if plot_title:
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

        ax.set_ylim(bottom=0.05, top=1.02)
        fig.savefig(fname=plot_output_path, dpi=150)
    else:
        return lls_accuracy, nlls_per_image
