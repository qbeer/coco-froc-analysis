import argparse
from froc.froc_curve import generate_froc_curve
from froc.bootstrap_curve import generate_bootstrap_curves


def run(args):
    if args.bootstrap:
        print("Generating bootstrap curves... (this may take a while)")
        generate_bootstrap_curves(
            args.gt_ann,
            args.pred_ann,
            args.n_bootstrap_samples,
            args.use_iou,
            args.iou_thres,
            args.plot_title,
            args.plot_output_path,
        )
    else:
        print("Generating single FROC curve...")
        generate_froc_curve(
            args.gt_ann,
            args.pred_ann,
            args.use_iou,
            args.iou_thres,
            args.plot_title,
            args.plot_output_path,
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
        "--n_bootstrap_samples",
        default=25,
        type=int,
        required=False,
        help="Number of bootstrap samples.",
    )
    parser.add_argument("--plot_title", type=str, required=True)
    parser.add_argument("--plot_output_path", type=str, required=True)

    args = parser.parse_args()

    run(args)
