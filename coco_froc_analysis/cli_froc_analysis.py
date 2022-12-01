from __future__ import annotations

import argparse

from .count import generate_bootstrap_count_curves
from .count import generate_count_curve
from .froc import generate_bootstrap_froc_curves
from .froc import generate_froc_curve


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bootstrap',
        type=int,
        default=1,
        help='Whether to do a single or bootstrap runs.',
    )
    parser.add_argument('--gt_ann', type=str, required=True)
    parser.add_argument('--pr_ann', type=str, required=True)
    parser.add_argument(
        '--use_iou',
        default=False,
        action='store_true',
        help='Use IoU score to decide based on `proximity`',
    )
    parser.add_argument(
        '--iou_thres',
        default=0.5,
        type=float,
        required=False,
        help='If IoU score is used the default threshold is set to .5',
    )
    parser.add_argument(
        '--n_sample_points',
        type=int,
        default=50,
        help='Number of points to evaluate the FROC curve at.',
    )
    parser.add_argument('--plot_title', type=str)
    parser.add_argument('--plot_output_path', type=str)

    parser.add_argument(
        '--test_ann',
        action='append',
        help='Extra ground-truth like annotations',
        required=False,
    )

    parser.add_argument(
        '--counts',
        default=False,
        action='store_true',
    )

    parser.add_argument(
        '--weighted',
        default=False,
        action='store_true',
    )

    args = parser.parse_args()

    if args.counts:
        if args.bootstrap > 1:
            generate_bootstrap_count_curves(
                gt_ann=args.gt_ann,
                pr_ann=args.pr_ann,
                n_bootstrap_samples=args.bootstrap,
                n_sample_points=args.n_sample_points,
                plot_title='Counts PR (bootstrap)' if args.plot_title is None else args.plot_title,
                plot_output_path='counts_bootstrap.png' if args.plot_output_path is None else args.plot_output_path,
                weighted=args.weighted,
                test_ann=args.test_ann,
            )
        else:
            generate_count_curve(
                gt_ann=args.gt_ann,
                pr_ann=args.pr_ann,
                weighted=args.weighted,
                plot_title='Counts PR' if args.plot_title is None else args.plot_title,
                plot_output_path='counts.png' if args.plot_output_path is None else args.plot_output_path,
                test_ann=args.test_ann,
            )

        exit(-1)

    if args.bootstrap > 1:
        print('Generating bootstrap curves... (this may take a while)')
        generate_bootstrap_froc_curves(
            gt_ann=args.gt_ann,
            pr_ann=args.pr_ann,
            n_bootstrap_samples=args.bootstrap,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title='FROC (bootstrap)' if args.plot_title is None else args.plot_title,
            plot_output_path='froc_bootstrap.png' if args.plot_output_path is None else args.plot_output_path,
            test_ann=args.test_ann,
        )
    else:
        print('Generating single FROC curve...')
        generate_froc_curve(
            gt_ann=args.gt_ann,
            pr_ann=args.pr_ann,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title='FROC' if args.plot_title is None else args.plot_title,
            plot_output_path='froc.png' if args.plot_output_path is None else args.plot_output_path,
            test_ann=args.test_ann,
        )
