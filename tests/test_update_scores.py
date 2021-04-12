import unittest
import numpy as np

from froc_analysis.utils import update_scores


class TestUpdateScore(unittest.TestCase):
    def setUp(self):
        self.json_data = [{"score": float(score) / 100.0} for score in range(0, 105, 5)]

    def test_thresholding(self):
        preds = update_scores(self.json_data, 0.5)
        self.assertEqual(len(preds), 10)
        preds = update_scores(self.json_data, 1.0)
        self.assertEqual(len(preds), 0)
