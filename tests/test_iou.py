import unittest
from froc_analysis.utils import get_iou_score


class TestIoU(unittest.TestCase):
    def setUp(self):
        self.gt_box = [100, 100, 100, 100]  # x, y, w, h
        self.pr_box1 = [100, 100, 50, 50]
        self.pr_box2 = [150, 150, 25, 25]
        self.pr_box3 = [200, 200, 100, 100]

    def test_calculate_overlapping_box_from_same_xy(self):
        iou = get_iou_score(self.gt_box, self.pr_box1)
        self.assertEqual(iou, 1 / 4)

    def test_calculate_overlapping_box_from_center(self):
        iou = get_iou_score(self.gt_box, self.pr_box2)
        self.assertEqual(iou, 1 / 16)

    def test_calculate_non_overlapping_boxes(self):
        iou = get_iou_score(self.gt_box, self.pr_box3)
        self.assertEqual(iou, 0.0)
