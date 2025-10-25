"""
Visualization utilities for embedding analysis and anomaly detection results.

Provides functions for creating:
- PCA and t-SNE scatter plots
- Mahalanobis distance distributions
- Chi-square probability histograms
- Confusion matrices
- Bayesian posterior probability curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, List, Tuple
import warnings

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def plot_embeddings_pca(embeddings: np.ndarray,
                        labels: np.ndarray,
                        title: str = "PCA Projection of Embeddings",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create 2D PCA scatter plot of embeddings colored by labels.
    
    Principal Component Analysis (PCA) finds the directions of maximum
    variance in high-dimensional data, allowing visualization in 2D.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with 0=normal, 1=anomaly
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate normal and anomalous points
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    # Plot normal points
    if np.any(normal_mask):
        ax.scatter(embeddings_2d[normal_mask, 0], 
                  embeddings_2d[normal_mask, 1],
                  c='blue', alpha=0.6, s=50, label='Normal', edgecolors='black', linewidths=0.5)
    
    # Plot anomalous points
    if np.any(anomaly_mask):
        ax.scatter(embeddings_2d[anomaly_mask, 0],
                  embeddings_2d[anomaly_mask, 1],
                  c='red', alpha=0.8, s=100, label='Anomalous', 
                  marker='X', edgecolors='black', linewidths=0.5)
    
    # Add labels and legend
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add explained variance text
    total_var = pca.explained_variance_ratio_[:2].sum()
    ax.text(0.02, 0.98, f'Total variance explained: {total_var:.1%}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_embeddings_tsne(embeddings: np.ndarray,
                         labels: np.ndarray,
                         title: str = "t-SNE Projection of Embeddings",
                         perplexity: int = 30,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Create 2D t-SNE scatter plot of embeddings colored by labels.
    
    t-SNE (t-Distributed Stochastic Neighbor Embedding) is better at
    preserving local structure and revealing clusters than PCA.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with 0=normal, 1=anomaly
        title: Plot title
        perplexity: t-SNE perplexity parameter (5-50 typical)
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Perform t-SNE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate normal and anomalous points
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    # Plot normal points
    if np.any(normal_mask):
        ax.scatter(embeddings_2d[normal_mask, 0],
                  embeddings_2d[normal_mask, 1],
                  c='blue', alpha=0.6, s=50, label='Normal', edgecolors='black', linewidths=0.5)
    
    # Plot anomalous points
    if np.any(anomaly_mask):
        ax.scatter(embeddings_2d[anomaly_mask, 0],
                  embeddings_2d[anomaly_mask, 1],
                  c='red', alpha=0.8, s=100, label='Anomalous',
                  marker='X', edgecolors='black', linewidths=0.5)
    
    # Add labels and legend
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_distance_distribution(distances: np.ndarray,
                               labels: np.ndarray,
                               threshold: Optional[float] = None,
                               title: str = "Mahalanobis Distance Distribution",
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot histogram of Mahalanobis distances separated by true labels.
    
    Args:
        distances: Array of Mahalanobis distances
        labels: True labels (0=normal, 1=anomaly)
        threshold: Optional detection threshold to display
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separate distances by label
    normal_distances = distances[labels == 0]
    anomaly_distances = distances[labels == 1]
    
    # Plot histograms
    bins = np.linspace(distances.min(), distances.max(), 50)
    
    if len(normal_distances) > 0:
        ax.hist(normal_distances, bins=bins, alpha=0.6, label='Normal',
               color='blue', edgecolor='black')
    
    if len(anomaly_distances) > 0:
        ax.hist(anomaly_distances, bins=bins, alpha=0.6, label='Anomalous',
               color='red', edgecolor='black')
    
    # Add threshold line
    if threshold is not None:
        ax.axvline(threshold, color='green', linestyle='--', linewidth=2,
                  label=f'Threshold = {threshold:.2f}')
    
    # Labels and legend
    ax.set_xlabel('Mahalanobis Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    if len(normal_distances) > 0 and len(anomaly_distances) > 0:
        stats_text = f'Normal: μ={normal_distances.mean():.2f}, σ={normal_distances.std():.2f}\n'
        stats_text += f'Anomaly: μ={anomaly_distances.mean():.2f}, σ={anomaly_distances.std():.2f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_chi2_probabilities(probabilities: np.ndarray,
                            labels: np.ndarray,
                            significance_level: float = 0.01,
                            title: str = "Chi-Square Probability Distribution",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot histogram of chi-square probabilities with significance threshold.
    
    Args:
        probabilities: Array of chi-square probabilities
        labels: True labels (0=normal, 1=anomaly)
        significance_level: Detection significance level (e.g., 0.01)
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separate probabilities by label
    normal_probs = probabilities[labels == 0]
    anomaly_probs = probabilities[labels == 1]
    
    # Plot histograms (log scale often better for probabilities)
    bins = np.logspace(np.log10(max(probabilities.min(), 1e-10)), 
                       np.log10(1.0), 50)
    
    if len(normal_probs) > 0:
        ax.hist(normal_probs, bins=bins, alpha=0.6, label='Normal',
               color='blue', edgecolor='black')
    
    if len(anomaly_probs) > 0:
        ax.hist(anomaly_probs, bins=bins, alpha=0.6, label='Anomalous',
               color='red', edgecolor='black')
    
    # Add significance threshold line
    ax.axvline(significance_level, color='green', linestyle='--', linewidth=2,
              label=f'Significance level = {significance_level}')
    
    # Labels and legend
    ax.set_xlabel('Chi-Square Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         labels: List[str] = ['Normal', 'Anomaly'],
                         title: str = "Confusion Matrix",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
               xticklabels=labels, yticklabels=labels, ax=ax,
               square=True, linewidths=1, linecolor='black')
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(y_true: np.ndarray,
                  y_scores: np.ndarray,
                  title: str = "ROC Curve",
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Args:
        y_true: True binary labels
        y_scores: Anomaly scores (higher = more anomalous)
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
           label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
           label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_covariance_heatmap(covariance_matrix: np.ndarray,
                           title: str = "Covariance Matrix Heatmap",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot heatmap of covariance matrix.
    
    Args:
        covariance_matrix: Square covariance matrix
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Only show first 50x50 if matrix is too large
    if covariance_matrix.shape[0] > 50:
        covariance_matrix = covariance_matrix[:50, :50]
        title += " (first 50 dimensions)"
    
    sns.heatmap(covariance_matrix, cmap='coolwarm', center=0, ax=ax,
               square=True, cbar_kws={'label': 'Covariance'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Dimension', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
