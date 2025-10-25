"""
Statistical anomaly detection using Mahalanobis distance and chi-square tests.

This module implements advanced linear algebra and probability-based methods
for detecting anomalous embeddings in high-dimensional space.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from typing import List, Tuple, Optional
import warnings


class MahalanobisDetector:
    """
    Anomaly detector using Mahalanobis distance and chi-square distribution.
    
    The Mahalanobis distance measures how far a point is from the center of
    a distribution, accounting for correlations between features. Under the
    assumption of multivariate normality, squared Mahalanobis distances
    follow a chi-square distribution.
    
    Attributes:
        mean_vector: Mean of the normal embeddings (centroid)
        covariance_matrix: Covariance matrix of normal embeddings
        inv_covariance_matrix: Inverse covariance matrix (precision matrix)
        dimension: Dimensionality of the embedding space
        chi2_threshold: Chi-square critical value for anomaly detection
    """
    
    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the Mahalanobis detector.
        
        Args:
            significance_level: Probability threshold for anomaly detection (default: 0.01)
                              Lower values = stricter detection
        """
        self.mean_vector = None
        self.covariance_matrix = None
        self.inv_covariance_matrix = None
        self.dimension = None
        self.chi2_threshold = None
        self.significance_level = significance_level
        self.is_fitted = False
        
    def fit(self, normal_embeddings: np.ndarray, regularization: float = 1e-6):
        """
        Fit the detector on normal (non-anomalous) embeddings.
        
        Computes the mean vector and covariance matrix from training data.
        Adds regularization to ensure covariance matrix is invertible.
        
        Args:
            normal_embeddings: Array of shape (n_samples, n_features)
            regularization: Small value added to diagonal for numerical stability
            
        Raises:
            ValueError: If embeddings have insufficient samples or dimensions
        """
        if len(normal_embeddings) < 2:
            raise ValueError("Need at least 2 samples to compute covariance")
        
        if normal_embeddings.shape[1] < 2:
            raise ValueError("Embeddings must have at least 2 dimensions")
        
        self.dimension = normal_embeddings.shape[1]
        
        # Compute mean vector (centroid of normal embeddings)
        self.mean_vector = np.mean(normal_embeddings, axis=0)
        
        # Compute covariance matrix
        self.covariance_matrix = np.cov(normal_embeddings, rowvar=False)
        
        # Add regularization to diagonal for numerical stability
        # This prevents singular matrix issues in high dimensions
        regularization_matrix = regularization * np.eye(self.dimension)
        self.covariance_matrix += regularization_matrix
        
        # Compute inverse covariance (precision matrix)
        try:
            self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        except np.linalg.LinAlgError:
            warnings.warn("Covariance matrix is singular. Using pseudo-inverse.")
            self.inv_covariance_matrix = np.linalg.pinv(self.covariance_matrix)
        
        # Calculate chi-square threshold
        # Squared Mahalanobis distances follow χ²(df=dimension) under normality
        self.chi2_threshold = stats.chi2.ppf(1 - self.significance_level, self.dimension)
        
        self.is_fitted = True
        
    def compute_mahalanobis_distance(self, embedding: np.ndarray) -> float:
        """
        Compute Mahalanobis distance for a single embedding.
        
        Formula: D_M(x) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
        where μ is the mean vector and Σ is the covariance matrix.
        
        Args:
            embedding: Single embedding vector of shape (n_features,)
            
        Returns:
            Mahalanobis distance (float)
            
        Raises:
            ValueError: If detector is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before computing distances")
        
        diff = embedding - self.mean_vector
        distance = np.sqrt(diff @ self.inv_covariance_matrix @ diff)
        return distance
    
    def compute_mahalanobis_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distances for multiple embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            Array of distances of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before computing distances")
        
        distances = np.array([
            self.compute_mahalanobis_distance(emb) for emb in embeddings
        ])
        return distances
    
    def compute_chi2_probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute chi-square probabilities for embeddings.
        
        Under the null hypothesis (embedding is normal), squared Mahalanobis
        distances follow χ²(df=dimension). This returns the probability that
        a point belongs to the normal distribution.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            Array of probabilities of shape (n_samples,)
            Higher probability = more likely to be normal
        """
        distances = self.compute_mahalanobis_distances(embeddings)
        squared_distances = distances ** 2
        
        # Compute survival function: P(X > squared_distance)
        # This is the probability of observing a distance this large or larger
        probabilities = 1 - stats.chi2.cdf(squared_distances, self.dimension)
        
        return probabilities
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict whether embeddings are anomalous.
        
        Uses chi-square test at the specified significance level.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            Binary array of shape (n_samples,) where 1 = anomaly, 0 = normal
        """
        probabilities = self.compute_chi2_probabilities(embeddings)
        predictions = (probabilities < self.significance_level).astype(int)
        return predictions
    
    def get_anomaly_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get continuous anomaly scores (Mahalanobis distances).
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            Array of scores of shape (n_samples,)
            Higher score = more anomalous
        """
        return self.compute_mahalanobis_distances(embeddings)
    
    def get_statistics(self) -> dict:
        """
        Get detector statistics and parameters.
        
        Returns:
            Dictionary with mean vector, covariance info, and thresholds
        """
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        eigenvalues = np.linalg.eigvals(self.covariance_matrix)
        
        return {
            'status': 'fitted',
            'dimension': self.dimension,
            'mean_vector': self.mean_vector,
            'covariance_determinant': np.linalg.det(self.covariance_matrix),
            'covariance_condition_number': np.linalg.cond(self.covariance_matrix),
            'covariance_eigenvalues_min': np.min(eigenvalues),
            'covariance_eigenvalues_max': np.max(eigenvalues),
            'chi2_threshold': self.chi2_threshold,
            'significance_level': self.significance_level
        }


def compute_covariance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix for embeddings.
    
    The covariance matrix Σ captures linear relationships between
    embedding dimensions: Σ_ij = E[(X_i - μ_i)(X_j - μ_j)]
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        
    Returns:
        Covariance matrix of shape (n_features, n_features)
    """
    return np.cov(embeddings, rowvar=False)


def compute_correlation_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix for embeddings.
    
    The correlation matrix normalizes covariances by standard deviations,
    giving values in [-1, 1].
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        
    Returns:
        Correlation matrix of shape (n_features, n_features)
    """
    return np.corrcoef(embeddings, rowvar=False)
