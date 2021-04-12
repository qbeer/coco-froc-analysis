import unittest
import os
import mock
import warnings

# remove tqdm logging while testing
def notqdm(iterable, *args, **kwargs):
    return iterable


from froc_analysis.froc_curve import froc_point
from froc_analysis import generate_froc_curve


class TestFrocCurve(unittest.TestCase):
    def setUp(self):
        self.gt_ann_path = f"{os.getcwd()}/tests/example_coco_data/face-mask.json"
        self.pr_prefect = (
            f"{os.getcwd()}/tests/example_coco_data/perfect_predictions.json"
        )
        self.pr_single_error_per_image_high_conf = f"{os.getcwd()}/tests/example_coco_data/single_error_per_image_predictions.json"
        self.pr_single_error_per_image_low_conf = f"{os.getcwd()}/tests/example_coco_data/single_error_per_image_low_confidence_predictions.json"
        warnings.simplefilter("ignore", category=UserWarning)

    def test_perfect_predictions_for_all_score_thresholds(self):
        for score_thresh in range(0, 100, 5):
            score_thresh = float(score_thresh) / 100.0
            stats = froc_point(
                self.gt_ann_path,
                self.pr_prefect,
                score_thresh,
                use_iou=False,
                iou_thres=0.0,
            )
            for cat_id in stats:
                self.assertEqual(stats[cat_id]["LL"] / stats[cat_id]["n_lesions"], 1.0)
                self.assertEqual(stats[cat_id]["NL"] / stats[cat_id]["n_images"], 0.0)

    def test_single_error_per_image_predictions_for_all_score_thresholds(self):
        for score_thresh in range(0, 100, 5):
            score_thresh = float(score_thresh) / 100.0
            stats = froc_point(
                self.gt_ann_path,
                self.pr_single_error_per_image_high_conf,
                score_thresh,
                use_iou=False,
                iou_thres=0.0,
            )
            for cat_id in stats:
                self.assertEqual(
                    stats[cat_id]["LL"] / stats[cat_id]["n_lesions"],
                    1.0 - stats[cat_id]["n_images"] / stats[cat_id]["n_lesions"],
                )

    def test_single_error_per_image_low_confidence_predictions_for_all_score_thresholds(
        self
    ):
        for score_thresh in range(0, 100, 5):
            score_thresh = float(score_thresh) / 100.0
            stats = froc_point(
                self.gt_ann_path,
                self.pr_single_error_per_image_low_conf,
                score_thresh,
                use_iou=False,
                iou_thres=0.0,
            )
            for cat_id in stats:
                specificity = (
                    1.0 - stats[cat_id]["n_images"] / stats[cat_id]["n_lesions"]
                )
                if score_thresh >= 0.25:
                    specificity = 0.0
                self.assertEqual(
                    stats[cat_id]["LL"] / stats[cat_id]["n_lesions"], specificity
                )

    @mock.patch("froc_analysis.froc_curve.tqdm", notqdm)
    def test_froc_curve_generation_for_perfect_score(self):
        lls, nlls = generate_froc_curve(
            self.gt_ann_path,
            self.pr_prefect,
            use_iou=False,
            iou_thres=0.0,
            n_sample_points=10,
            plot_title=None,
            plot_output_path=None,
        )
        self.assertListEqual(lls[8], [1.0 for _ in range(10)])
        self.assertListEqual(nlls[8], [0.0 for _ in range(10)])

    @mock.patch("froc_analysis.froc_curve.tqdm", notqdm)
    def test_froc_curve_generation_for_single_error_per_image(self):
        lls, nlls = generate_froc_curve(
            self.gt_ann_path,
            self.pr_single_error_per_image_high_conf,
            use_iou=False,
            iou_thres=0.0,
            n_sample_points=10,
            plot_title=None,
            plot_output_path=None,
        )
        self.assertListEqual(lls[8], [0.7 for _ in range(10)])
        self.assertListEqual(nlls[8], [0.0 for _ in range(10)])

    @mock.patch("froc_analysis.froc_curve.tqdm", notqdm)
    def test_froc_curve_generation_for_single_error_per_image_low_confidence(self):
        lls, nlls = generate_froc_curve(
            self.gt_ann_path,
            self.pr_single_error_per_image_low_conf,
            use_iou=False,
            iou_thres=0.0,
            n_sample_points=10,
            plot_title=None,
            plot_output_path=None,
        )
        self.assertListEqual(lls[8], [0.7 for _ in range(3)] + [0.0 for _ in range(7)])
        self.assertListEqual(nlls[8], [0.0 for _ in range(10)])

    @mock.patch("froc_analysis.froc_curve.tqdm", notqdm)
    def test_froc_curve_saving_to_plot(self):
        generate_froc_curve(
            self.gt_ann_path,
            self.pr_prefect,
            use_iou=False,
            iou_thres=0.0,
            n_sample_points=10,
            plot_title="perfect scores",
            plot_output_path="/tmp/froc_perfect_scores.png",
        )
