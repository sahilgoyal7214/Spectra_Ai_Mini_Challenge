"""
Bayesian posterior probability analysis for anomaly detection.

This module implements Bayesian reasoning to update beliefs about whether
a prompt is anomalous based on detector outputs and prior knowledge.
"""

import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class BayesianAnomalyAnalyzer:
    """
    Bayesian analyzer for computing posterior probabilities of anomalies.
    
    Uses Bayes' theorem to combine:
    - Prior probability of anomalies (base rate)
    - Likelihood of detector flagging normal prompts (False Positive Rate)
    - Likelihood of detector flagging anomalous prompts (True Positive Rate)
    
    Formula: P(anomaly | flagged) = P(flagged | anomaly) * P(anomaly) / P(flagged)
    """
    
    def __init__(self, 
                 prior_anomaly_rate: float = 0.1,
                 true_positive_rate: float = 0.95,
                 false_positive_rate: float = 0.05):
        """
        Initialize Bayesian analyzer with detection parameters.
        
        Args:
            prior_anomaly_rate: Prior probability P(anomaly) - base rate in population
            true_positive_rate: P(flagged | anomaly) - detector sensitivity/recall
            false_positive_rate: P(flagged | normal) - detector false alarm rate
            
        Example:
            If 10% of prompts are truly anomalous (prior=0.1), and detector
            correctly flags 95% of anomalies (TPR=0.95) but also incorrectly
            flags 5% of normal prompts (FPR=0.05), then:
            >>> analyzer = BayesianAnomalyAnalyzer(0.1, 0.95, 0.05)
        """
        self.prior_anomaly_rate = prior_anomaly_rate
        self.true_positive_rate = true_positive_rate
        self.false_positive_rate = false_positive_rate
        
        # Derived probabilities
        self.prior_normal_rate = 1 - prior_anomaly_rate
        self.true_negative_rate = 1 - false_positive_rate
        self.false_negative_rate = 1 - true_positive_rate
        
    def compute_posterior_anomaly(self, flagged: bool = True) -> float:
        """
        Compute posterior probability that a prompt is anomalous.
        
        Uses Bayes' theorem:
        P(anomaly | flagged) = P(flagged | anomaly) * P(anomaly) / P(flagged)
        
        where P(flagged) = P(flagged | anomaly)*P(anomaly) + P(flagged | normal)*P(normal)
        
        Args:
            flagged: Whether the detector flagged this prompt (default: True)
            
        Returns:
            Posterior probability P(anomaly | observation)
        """
        if flagged:
            # Likelihood of being flagged given anomaly
            likelihood_anomaly = self.true_positive_rate
            # Likelihood of being flagged given normal
            likelihood_normal = self.false_positive_rate
        else:
            # Likelihood of NOT being flagged given anomaly
            likelihood_anomaly = self.false_negative_rate
            # Likelihood of NOT being flagged given normal
            likelihood_normal = self.true_negative_rate
        
        # Bayes' theorem numerator
        numerator = likelihood_anomaly * self.prior_anomaly_rate
        
        # Bayes' theorem denominator (total probability)
        denominator = (likelihood_anomaly * self.prior_anomaly_rate + 
                      likelihood_normal * self.prior_normal_rate)
        
        if denominator == 0:
            return 0.0
        
        posterior = numerator / denominator
        return posterior
    
    def compute_posterior_normal(self, flagged: bool = True) -> float:
        """
        Compute posterior probability that a prompt is normal.
        
        Args:
            flagged: Whether the detector flagged this prompt
            
        Returns:
            Posterior probability P(normal | observation)
        """
        return 1 - self.compute_posterior_anomaly(flagged)
    
    def compute_likelihood_ratio(self) -> float:
        """
        Compute likelihood ratio for positive detection.
        
        LR = P(flagged | anomaly) / P(flagged | normal) = TPR / FPR
        
        This measures how much more likely a flagged result is under
        the anomaly hypothesis vs. the normal hypothesis.
        
        Returns:
            Likelihood ratio (positive values favor anomaly hypothesis)
        """
        if self.false_positive_rate == 0:
            return float('inf')
        
        return self.true_positive_rate / self.false_positive_rate
    
    def compute_bayes_factor(self, flagged: bool = True) -> float:
        """
        Compute Bayes factor comparing anomaly to normal hypothesis.
        
        BF = P(data | anomaly) / P(data | normal)
        
        Interpretation:
        - BF > 10: Strong evidence for anomaly
        - BF 3-10: Moderate evidence for anomaly  
        - BF 1-3: Weak evidence for anomaly
        - BF < 1: Evidence against anomaly
        
        Args:
            flagged: Whether the detector flagged this prompt
            
        Returns:
            Bayes factor
        """
        if flagged:
            numerator = self.true_positive_rate
            denominator = self.false_positive_rate
        else:
            numerator = self.false_negative_rate
            denominator = self.true_negative_rate
        
        if denominator == 0:
            return float('inf') if numerator > 0 else 1.0
        
        return numerator / denominator
    
    def update_from_multiple_detectors(self, 
                                      detectors_flagged: List[bool],
                                      tpr_list: Optional[List[float]] = None,
                                      fpr_list: Optional[List[float]] = None) -> float:
        """
        Update posterior probability using multiple independent detectors.
        
        Applies Bayes' theorem sequentially, using each detector's output
        to refine the probability estimate.
        
        Args:
            detectors_flagged: List of boolean flags from each detector
            tpr_list: Optional list of TPRs for each detector (uses default if None)
            fpr_list: Optional list of FPRs for each detector (uses default if None)
            
        Returns:
            Final posterior probability after all detectors
        """
        posterior = self.prior_anomaly_rate
        
        for i, flagged in enumerate(detectors_flagged):
            # Use detector-specific rates if provided, else use defaults
            tpr = tpr_list[i] if tpr_list else self.true_positive_rate
            fpr = fpr_list[i] if fpr_list else self.false_positive_rate
            
            # Compute likelihood for this detector
            if flagged:
                likelihood_anomaly = tpr
                likelihood_normal = fpr
            else:
                likelihood_anomaly = 1 - tpr
                likelihood_normal = 1 - fpr
            
            # Update posterior using current detector
            numerator = likelihood_anomaly * posterior
            denominator = (likelihood_anomaly * posterior + 
                         likelihood_normal * (1 - posterior))
            
            if denominator > 0:
                posterior = numerator / denominator
        
        return posterior
    
    def plot_posterior_vs_prior(self, 
                               prior_range: Tuple[float, float] = (0.001, 0.999),
                               n_points: int = 100,
                               flagged: bool = True) -> plt.Figure:
        """
        Plot how posterior probability changes with different priors.
        
        Shows sensitivity of conclusions to prior beliefs.
        
        Args:
            prior_range: Range of prior probabilities to plot
            n_points: Number of points in the plot
            flagged: Whether detector flagged the prompt
            
        Returns:
            Matplotlib figure
        """
        priors = np.linspace(prior_range[0], prior_range[1], n_points)
        posteriors = []
        
        for prior in priors:
            temp_analyzer = BayesianAnomalyAnalyzer(
                prior_anomaly_rate=prior,
                true_positive_rate=self.true_positive_rate,
                false_positive_rate=self.false_positive_rate
            )
            posterior = temp_analyzer.compute_posterior_anomaly(flagged)
            posteriors.append(posterior)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(priors, posteriors, 'b-', linewidth=2, label='Posterior')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Prior = Posterior (no update)')
        
        # Mark current configuration
        current_posterior = self.compute_posterior_anomaly(flagged)
        ax.plot(self.prior_anomaly_rate, current_posterior, 'ro', 
               markersize=10, label=f'Current (prior={self.prior_anomaly_rate:.2f})')
        
        ax.set_xlabel('Prior P(anomaly)', fontsize=12)
        ax.set_ylabel('Posterior P(anomaly | flagged)', fontsize=12)
        ax.set_title(f'Bayesian Update: TPR={self.true_positive_rate}, FPR={self.false_positive_rate}',
                    fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        return fig
    
    def get_summary(self) -> dict:
        """
        Get summary of Bayesian analysis parameters and results.
        
        Returns:
            Dictionary with priors, likelihoods, and posteriors
        """
        posterior_if_flagged = self.compute_posterior_anomaly(flagged=True)
        posterior_if_not_flagged = self.compute_posterior_anomaly(flagged=False)
        likelihood_ratio = self.compute_likelihood_ratio()
        bayes_factor = self.compute_bayes_factor(flagged=True)
        
        return {
            'prior_anomaly_rate': self.prior_anomaly_rate,
            'true_positive_rate': self.true_positive_rate,
            'false_positive_rate': self.false_positive_rate,
            'posterior_if_flagged': posterior_if_flagged,
            'posterior_if_not_flagged': posterior_if_not_flagged,
            'likelihood_ratio': likelihood_ratio,
            'bayes_factor': bayes_factor,
            'interpretation': self._interpret_bayes_factor(bayes_factor)
        }
    
    @staticmethod
    def _interpret_bayes_factor(bf: float) -> str:
        """Interpret Bayes factor value."""
        if bf > 100:
            return "Decisive evidence for anomaly"
        elif bf > 30:
            return "Very strong evidence for anomaly"
        elif bf > 10:
            return "Strong evidence for anomaly"
        elif bf > 3:
            return "Moderate evidence for anomaly"
        elif bf > 1:
            return "Weak evidence for anomaly"
        elif bf == 1:
            return "No evidence either way"
        else:
            return "Evidence against anomaly"


def compute_posterior_probability(prior: float, 
                                 likelihood_anomaly: float, 
                                 likelihood_normal: float) -> float:
    """
    Simple Bayes' theorem computation.
    
    P(anomaly | data) = P(data | anomaly) * P(anomaly) / P(data)
    
    Args:
        prior: P(anomaly)
        likelihood_anomaly: P(data | anomaly)
        likelihood_normal: P(data | normal)
        
    Returns:
        Posterior probability P(anomaly | data)
    """
    numerator = prior * likelihood_anomaly
    denominator = numerator + (1 - prior) * likelihood_normal
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator
