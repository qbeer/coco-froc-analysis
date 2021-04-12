import unittest

from froc_analysis.utils import build_pr_id2annotations, build_gt_id2annotations


class TestId2Annotation(unittest.TestCase):
    def setUp(self):
        self.gt = {
            "images": [{"id": 1}, {"id": 2}],
            "annotations": [
                {"image_id": 1},
                {"image_id": 1},
                {"image_id": 1},
                {"image_id": 2},
            ],
        }
        self.pr = [
            {"image_id": 1},
            {"image_id": 1},
            {"image_id": 1},
            {"image_id": 1},
            {"image_id": 1},
            {"image_id": 2},
            {"image_id": 2},
            {"image_id": 2},
        ]

    def test_build_ground_truth_id2annotations_dict(self):
        id2annotation = build_gt_id2annotations(self.gt)
        self.assertEqual(len(id2annotation[1]), 3)
        self.assertEqual(len(id2annotation[2]), 1)

    def test_build_prediction_id2annotations_dict(self):
        id2annotation = build_pr_id2annotations(self.pr)
        self.assertEqual(len(id2annotation[1]), 5)
        self.assertEqual(len(id2annotation[2]), 3)
