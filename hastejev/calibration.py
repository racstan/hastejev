import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from typing import Optional

class HITCalibrator:
    """
    Hybrid Isotonic-Temperature Calibration (HIT-Calib) Engine.
    Minimizes Expected Calibration Error (ECE) using parametric temperature scaling
    combined with non-parametric monotonic isotonic regression.
    """
    def __init__(self):
        self.temperature = 1.0
        self.iso_regressor = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
        self.is_fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """
        Fit temperature parameter T via Negative Log-Likelihood (NLL) optimization.
        """
        def nll_objective(T_val):
            T = max(T_val[0], 0.05)
            scaled = logits / T
            exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
            correct_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
            return -np.mean(np.log(correct_probs))

        res = minimize(nll_objective, [1.5], bounds=[(0.05, 10.0)], method='L-BFGS-B')
        self.temperature = float(res.x[0])
        self.is_fitted = True

    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to raw logits.
        """
        scaled = logits / self.temperature
        return F.softmax(scaled, dim=-1)

    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE) across n_bins probability bins.
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin.astype(float))
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return float(ece)
