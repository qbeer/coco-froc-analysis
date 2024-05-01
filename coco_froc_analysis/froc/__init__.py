from __future__ import annotations

from .bootstrap_froc_curve import generate_bootstrap_froc_curves  # noqa: F401, E501
from .froc_curve import froc_point  # noqa: F401
from .froc_curve import generate_froc_curve  # noqa: F401

documentation = None
"""
This submodule provides functionality for evaluating Free-Response Receiver Operating Characteristic (FROC) curves.
It includes functions for running bootstrapping, generating FROC curves, calculating statistics for FROC points,
and updating FROC statistics.

Functions:
- run_bootstrap: Runs a single iteration of bootstrapping to generate ground truth annotations and predictions for
                 calculating FROC curves.
- generate_bootstrap_froc_curves: Generates bootstrapped FROC curves based on ground truth and predicted annotations.
- generate_froc_curve: Generates a FROC curve based on ground truth annotations and predicted annotations.
- froc_point: Calculates statistics for a single point on the FROC curve based on ground truth and predicted annotations.
- calc_scores: Calculates Sensitivity and False Positive Rate (FPR) based on FROC statistics.
- init_stats: Initializes statistics before counting lesion and non-lesion localizations.
- update_stats: Updates statistics as going through images of the dataset.


Note: This submodule requires JSON formatted ground truth and predicted annotation files for evaluation.
"""
