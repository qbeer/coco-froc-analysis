import unittest
import os
import warnings

from froc_analysis.count_curve import count_point


class TestFrocCurve(unittest.TestCase):
    def setUp(self):
        self.gt_ann_path = f"{os.getcwd()}/tests/example_coco_data/face-mask.json"
        self.pr_prefect = (
            f"{os.getcwd()}/tests/example_coco_data/perfect_predictions.json")
        self.pr_single_error_per_image_high_conf = f"{os.getcwd()}/tests/example_coco_data/single_error_per_image_predictions.json"
        self.pr_single_error_per_image_low_conf = f"{os.getcwd()}/tests/example_coco_data/single_error_per_image_low_confidence_predictions.json"
        warnings.simplefilter("ignore", category=UserWarning)

    def test_high_conf_single_error_predictions_for_all_score_thresholds(self):
        for score_thresh in range(0, 100, 5):
            score_thresh = float(score_thresh) / 100.0
            stats = count_point(self.gt_ann_path,
                                self.pr_single_error_per_image_high_conf,
                                score_thresh,
                                weighted=False)

            for cat_id in stats:
                self.assertEqual(stats[cat_id]["FN"],
                                 3)  # 3 is the number of images
                self.assertEqual(stats[cat_id]["P"] - 3, stats[cat_id]['TP'])
