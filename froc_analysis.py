import argparse
from froc_analysis import generate_froc_curve
from froc_analysis import generate_bootstrap_curves


def run(args):
    if args.bootstrap:
        print("Generating bootstrap curves... (this may take a while)")
        generate_bootstrap_curves(
            gt_ann=args.gt_ann,
            pr_ann=args.pred_ann,
            n_bootstrap_samples=args.n_bootstrap_samples,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title=args.plot_title,
            plot_output_path=args.plot_output_path
        )
    else:
        print("Generating single FROC curve...")
        generate_froc_curve(
            gt_ann=args.gt_ann,
            pr_ann=args.pred_ann,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title=args.plot_title,
            plot_output_path=args.plot_output_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Whether to do a single or bootstrap runs.",
    )
    parser.add_argument("--gt_ann", type=str, required=True)
    parser.add_argument("--pred_ann", type=str, required=True)
    parser.add_argument(
        "--use_iou",
        default=False,
        action="store_true",
        help="Use IoU score to decide on `proximity` rather then using center pixel inside GT box.",
    )
    parser.add_argument(
        "--iou_thres",
        default=0.5,
        type=float,
        required=False,
        help="If IoU score is used the default threshold is arbitrarily set to .5",
    )
    parser.add_argument(
        "--n_sample_points",
        type=int,
        default=50,
        help='Number of points to evaluate the FROC curve at.'
    )
    parser.add_argument(
        "--n_bootstrap_samples",
        default=25,
        type=int,
        required=False,
        help="Number of bootstrap samples.",
    )
    parser.add_argument("--plot_title", type=str, default='FROC')
    parser.add_argument("--plot_output_path", type=str, default='froc.png')

    args = parser.parse_args()

    run(args)
