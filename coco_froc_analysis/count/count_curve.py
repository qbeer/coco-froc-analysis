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
from .count_stats import init_stats
from .count_stats import update_stats


def count_point(gt_ann, pr_ann, score_thres, weighted):
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
):
    precision = {}
    recall = {}

    for score_thres in tqdm(
            np.linspace(0.0, 1.0, n_sample_points, endpoint=False),
    ):
        stats = count_point(gt_ann, pr_ann, score_thres, weighted)
        precision, recall = calc_scores(stats, precision, recall)

    if plot_title:
        fig, ax = plt.subplots(figsize=[20, 9])
        ins = ax.inset_axes([0.05, 0.05, 0.45, 0.4])
        ins.set_xlim([0.65, 1.0])
        ins.set_xticks([.7, .75, .8, .85, .9, .95], fontsize=30)

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
                for t_ann, c in zip(test_ann, colors):
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
                            x=np.min(rec), y=_precision[category_id][0] - 0.035,
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
        ax.set_xlim(np.min(rec), 1)

        fig.savefig(plot_output_path, dpi=150)
    else:
        return precision, recall
