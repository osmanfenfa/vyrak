"""
zelvryn/ml/evaluator.py
========================
ModelEvaluator — computes comprehensive evaluation metrics for Zelvryn's
ML classifiers, supporting both batch evaluation and threshold sweep analysis.

Why security-specific metrics matter
-------------------------------------
Standard ML metrics (accuracy, F1) are insufficient for security classifiers.
A system that correctly identifies 99% of sessions as human may still miss
99% of attacks if bot sessions are rare. Security systems require:

  FAR (False Acceptance Rate)
    = False Negatives / Total Bots
    = fraction of bot sessions incorrectly allowed through
    A FAR of 0.05 means 5% of actual bot attacks are missed.
    MINIMISE this — missing an attack is the primary security risk.

  FRR (False Rejection Rate)
    = False Positives / Total Humans
    = fraction of legitimate user sessions incorrectly challenged/blocked
    An FRR of 0.02 means 2% of real users get a false challenge.
    MINIMISE this — too many false positives destroy user experience.

  EER (Equal Error Rate)
    The threshold at which FAR == FRR. Lower EER = better classifier.
    Used as a single-number summary for classifier comparison.

Threshold sweep
---------------
The evaluator sweeps P(bot) thresholds from 0.0 to 1.0 and computes FAR/FRR
at each point. This produces the DET (Detection Error Tradeoff) curve and
allows operators to choose a threshold that balances security and usability
for their specific deployment context.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.classifier import RandomForestClassifier
from core.hoeffding import HoeffdingTreeClassifier
from core.constants import ML_BOT_CONFIDENCE_THRESH, ML_FEATURE_DIM
from core.features import FeatureExtractor

logger = logging.getLogger(__name__)

# Number of threshold points for the ROC/DET curve sweep
THRESHOLD_SWEEP_POINTS: int = 100


@dataclass
class EvaluationReport:
    """
    Comprehensive evaluation report for a single classifier.

    Attributes
    ----------
    classifier_version : str
    n_samples : int
    n_human : int
    n_bot : int
    threshold_used : float
        P(bot) threshold used for binary predictions.
    precision : float
    recall : float
    f1 : float
    accuracy : float
    far : float
        False Acceptance Rate (bot sessions incorrectly allowed).
    frr : float
        False Rejection Rate (human sessions incorrectly blocked).
    eer : float
        Equal Error Rate from the threshold sweep.
    eer_threshold : float
        P(bot) threshold at which EER occurs.
    confusion_matrix : dict
        {TP, FP, TN, FN} counts.
    roc_points : list of dict
        [{threshold, far, frr, tpr, fpr}, ...] for plotting curves.
    top_features : list of tuple
        (feature_name, importance) top 5.
    evaluated_at : float
    """
    classifier_version:  str
    n_samples:           int                   = 0
    n_human:             int                   = 0
    n_bot:               int                   = 0
    threshold_used:      float                 = ML_BOT_CONFIDENCE_THRESH
    precision:           float                 = 0.0
    recall:              float                 = 0.0
    f1:                  float                 = 0.0
    accuracy:            float                 = 0.0
    far:                 float                 = 0.0
    frr:                 float                 = 0.0
    eer:                 float                 = 0.0
    eer_threshold:       float                 = 0.5
    confusion_matrix:    Dict[str, int]        = field(default_factory=dict)
    roc_points:          List[Dict[str, float]] = field(default_factory=list)
    top_features:        List[Tuple[str,float]] = field(default_factory=list)
    evaluated_at:        float                 = field(default_factory=time.time)

    def passes_quality_gate(
        self,
        max_far: float = 0.10,
        max_frr: float = 0.05,
        min_f1:  float = 0.80,
    ) -> Tuple[bool, List[str]]:
        """
        Check whether this model meets promotion quality gates.

        Parameters
        ----------
        max_far : float
            Maximum acceptable False Acceptance Rate.
        max_frr : float
            Maximum acceptable False Rejection Rate.
        min_f1 : float
            Minimum acceptable F1 score.

        Returns
        -------
        tuple of (bool, list of str)
            (passes, list_of_failed_criteria)
        """
        failures = []
        if self.far > max_far:
            failures.append(f"FAR={self.far:.3f} > max_far={max_far}")
        if self.frr > max_frr:
            failures.append(f"FRR={self.frr:.3f} > max_frr={max_frr}")
        if self.f1 < min_f1:
            failures.append(f"F1={self.f1:.3f} < min_f1={min_f1}")
        return (len(failures) == 0, failures)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classifier_version": self.classifier_version,
            "n_samples":          self.n_samples,
            "n_human":            self.n_human,
            "n_bot":              self.n_bot,
            "threshold_used":     round(self.threshold_used, 4),
            "precision":          round(self.precision, 4),
            "recall":             round(self.recall, 4),
            "f1":                 round(self.f1, 4),
            "accuracy":           round(self.accuracy, 4),
            "far":                round(self.far, 4),
            "frr":                round(self.frr, 4),
            "eer":                round(self.eer, 4),
            "eer_threshold":      round(self.eer_threshold, 4),
            "confusion_matrix":   self.confusion_matrix,
            "top_features":       [(n, round(i, 4)) for n, i in self.top_features],
            "evaluated_at":       self.evaluated_at,
        }

    def summary(self) -> str:
        return (
            f"EvaluationReport({self.classifier_version}: "
            f"n={self.n_samples} "
            f"F1={self.f1:.3f} "
            f"FAR={self.far:.3f} "
            f"FRR={self.frr:.3f} "
            f"EER={self.eer:.3f}@{self.eer_threshold:.2f})"
        )


class ModelEvaluator:
    """
    Evaluates ML classifier performance using security-appropriate metrics.

    Parameters
    ----------
    threshold : float, optional
        P(bot) decision threshold. Default ML_BOT_CONFIDENCE_THRESH.
    sweep_points : int, optional
        Number of threshold values to evaluate for ROC/DET curves.
    """

    def __init__(
        self,
        threshold:    float = ML_BOT_CONFIDENCE_THRESH,
        sweep_points: int   = THRESHOLD_SWEEP_POINTS,
    ) -> None:
        self.threshold    = threshold
        self.sweep_points = sweep_points

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMARY EVALUATION
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        classifier: RandomForestClassifier,
        X: List[List[float]],
        y: List[int],
        version: str = "unknown",
    ) -> EvaluationReport:
        """
        Run a full evaluation of a classifier on a labeled dataset.

        Computes:
          - Binary classification metrics at the configured threshold
          - Confusion matrix (TP, FP, TN, FN)
          - FAR and FRR at the configured threshold
          - Full threshold sweep for DET/ROC curve
          - EER from the sweep
          - Feature importances from the classifier

        Parameters
        ----------
        classifier : RandomForestClassifier
        X : list of list of float
        y : list of int
        version : str, optional
            Version label for the report.

        Returns
        -------
        EvaluationReport
        """
        if not classifier.is_trained:
            logger.warning("Evaluating an untrained classifier — all metrics will be 0.")
            return EvaluationReport(classifier_version=version)

        logger.info(
            "Evaluating classifier=%s on %d samples (%d human, %d bot)",
            version, len(X), y.count(0), y.count(1)
        )

        # Get probability scores for all samples
        probas = classifier.predict_batch(X)

        # Binary predictions at configured threshold
        y_pred = [1 if p >= self.threshold else 0 for p in probas]

        # Core metrics
        cm       = self._confusion_matrix(y, y_pred)
        prec     = self._precision(cm)
        rec      = self._recall(cm)
        f1       = self._f1(prec, rec)
        acc      = self._accuracy(y, y_pred)
        far, frr = self._far_frr(y, y_pred)

        # Threshold sweep for DET curve
        roc_points      = self._sweep_thresholds(probas, y)
        eer, eer_thresh = self._equal_error_rate(roc_points)

        # Feature importances
        top_features = classifier.get_top_features(5)

        report = EvaluationReport(
            classifier_version = version,
            n_samples          = len(X),
            n_human            = y.count(0),
            n_bot              = y.count(1),
            threshold_used     = self.threshold,
            precision          = prec,
            recall             = rec,
            f1                 = f1,
            accuracy           = acc,
            far                = far,
            frr                = frr,
            eer                = eer,
            eer_threshold      = eer_thresh,
            confusion_matrix   = cm,
            roc_points         = roc_points,
            top_features       = top_features,
        )

        logger.info(report.summary())
        return report

    def compare(
        self,
        clf_a: RandomForestClassifier,
        clf_b: RandomForestClassifier,
        X: List[List[float]],
        y: List[int],
        version_a: str = "model_a",
        version_b: str = "model_b",
    ) -> Dict[str, Any]:
        """
        Compare two classifiers on the same dataset.

        Returns a comparison dict showing which model wins on each metric
        and by how much.

        Parameters
        ----------
        clf_a, clf_b : RandomForestClassifier
        X : list of list of float
        y : list of int
        version_a, version_b : str

        Returns
        -------
        dict
        """
        report_a = self.evaluate(clf_a, X, y, version_a)
        report_b = self.evaluate(clf_b, X, y, version_b)

        def winner(val_a: float, val_b: float, higher_is_better: bool) -> str:
            if higher_is_better:
                return version_a if val_a > val_b else version_b
            else:
                return version_a if val_a < val_b else version_b

        return {
            "version_a":   version_a,
            "version_b":   version_b,
            "report_a":    report_a.to_dict(),
            "report_b":    report_b.to_dict(),
            "winners": {
                "f1":      winner(report_a.f1,  report_b.f1,  True),
                "far":     winner(report_a.far, report_b.far, False),
                "frr":     winner(report_a.frr, report_b.frr, False),
                "eer":     winner(report_a.eer, report_b.eer, False),
                "overall": (
                    version_a
                    if (report_a.f1 > report_b.f1
                        and report_a.far < report_b.far)
                    else version_b
                ),
            },
            "deltas": {
                "f1":  round(report_a.f1  - report_b.f1,  4),
                "far": round(report_a.far - report_b.far, 4),
                "frr": round(report_a.frr - report_b.frr, 4),
                "eer": round(report_a.eer - report_b.eer, 4),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # METRIC CALCULATIONS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
        """
        Compute confusion matrix counts.

        Convention (bot = positive class):
          TP = correctly predicted bot
          FP = human incorrectly predicted as bot  (causes FRR)
          TN = correctly predicted human
          FN = bot incorrectly predicted as human  (causes FAR)
        """
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

    @staticmethod
    def _precision(cm: Dict[str, int]) -> float:
        denom = cm["TP"] + cm["FP"]
        return cm["TP"] / denom if denom > 0 else 0.0

    @staticmethod
    def _recall(cm: Dict[str, int]) -> float:
        denom = cm["TP"] + cm["FN"]
        return cm["TP"] / denom if denom > 0 else 0.0

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        denom = precision + recall
        return 2 * precision * recall / denom if denom > 0 else 0.0

    @staticmethod
    def _accuracy(y_true: List[int], y_pred: List[int]) -> float:
        if not y_true:
            return 0.0
        return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    @staticmethod
    def _far_frr(y_true: List[int], y_pred: List[int]) -> Tuple[float, float]:
        """
        Compute FAR and FRR.

        FAR = FN / (FN + TP) = missed bots / all bots
        FRR = FP / (FP + TN) = blocked humans / all humans
        """
        total_bots   = sum(1 for t in y_true if t == 1)
        total_humans = sum(1 for t in y_true if t == 0)

        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)

        far = fn / total_bots   if total_bots   > 0 else 0.0
        frr = fp / total_humans if total_humans > 0 else 0.0
        return (far, frr)

    def _sweep_thresholds(
        self,
        probas: List[float],
        y_true: List[int],
    ) -> List[Dict[str, float]]:
        """
        Sweep P(bot) thresholds from 0.0 to 1.0 and compute FAR/FRR at each.

        Parameters
        ----------
        probas : list of float
            P(bot) scores from the classifier.
        y_true : list of int
            Ground-truth labels.

        Returns
        -------
        list of dict
            Each dict: {threshold, far, frr, tpr, fpr}
        """
        thresholds = [i / self.sweep_points for i in range(self.sweep_points + 1)]
        points = []

        for t in thresholds:
            y_pred = [1 if p >= t else 0 for p in probas]
            far, frr = self._far_frr(y_true, y_pred)

            # TPR = Recall = 1 - FAR (bots correctly detected)
            # FPR = 1 - Specificity = FRR (humans incorrectly flagged)
            points.append({
                "threshold": round(t, 4),
                "far":       round(far, 4),
                "frr":       round(frr, 4),
                "tpr":       round(1.0 - far, 4),
                "fpr":       round(frr, 4),
            })

        return points

    @staticmethod
    def _equal_error_rate(
        roc_points: List[Dict[str, float]],
    ) -> Tuple[float, float]:
        """
        Find the Equal Error Rate (where FAR ≈ FRR) from the sweep.

        Parameters
        ----------
        roc_points : list of dict

        Returns
        -------
        tuple of (float, float)
            (eer_value, threshold_at_eer)
        """
        best_diff  = float("inf")
        best_eer   = 0.5
        best_thresh = 0.5

        for pt in roc_points:
            diff = abs(pt["far"] - pt["frr"])
            if diff < best_diff:
                best_diff   = diff
                best_eer    = (pt["far"] + pt["frr"]) / 2.0
                best_thresh = pt["threshold"]

        return (round(best_eer, 4), round(best_thresh, 4))

    def __repr__(self) -> str:
        return (
            f"ModelEvaluator("
            f"threshold={self.threshold}, "
            f"sweep_points={self.sweep_points})"
        )
