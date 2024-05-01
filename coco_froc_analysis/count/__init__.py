from __future__ import annotations

from .bootstrap_count_curve import generate_bootstrap_count_curves  # noqa: F401, E501
from .count_curve import count_point  # noqa: F401
from .count_curve import generate_count_curve  # noqa: F401

documentation = None


"""
This submodule provides functionality for generating count curves, computing count statistics,
and generating bootstrapped count curves.

Functions:
- generate_bootstrap_count_curves: Generates bootstrapped count curves, which are statistical estimates of precision-recall curves
                                   obtained by resampling the data with replacement.
- generate_count_curve: Generates a count curve based on ground truth annotations and predicted annotations.
- count_point: Computes statistics based on ground truth annotations and predicted annotations.
- calc_scores: Calculates precision and recall based on count statistics.
- init_stats: Initializes statistics for each category based on ground truth annotations.
- update_stats: Updates statistics based on ground truth and predicted annotations.
"""
