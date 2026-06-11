import random
import numpy as np
import torch


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# NORMALIZATION
# =========================
def normalize_features(X):
    """
    Standard normalization (mean=0, std=1)
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std


def min_max_scaling(X):
    """
    Scale features between 0 and 1
    """
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8)


# =========================
# DISTANCE FUNCTIONS
# =========================
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def pairwise_distances(coords):
    """
    Compute full distance matrix (can be expensive)
    """
    n = coords.shape[0]
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = euclidean_distance(coords[i], coords[j])

    return dist_matrix


# =========================
# LOGGING
# =========================
def print_section(title):
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50 + "\n")


def print_step(message):
    print(f"   → {message}")


# =========================
# MASK UTILITIES
# =========================
def create_missing_mask(X):
    """
    1 = observed, 0 = missing
    """
    return (~np.isnan(X)).astype(float)


def apply_mask_loss(diff, mask):
    """
    Apply mask to loss computation
    """
    return (diff * mask).sum() / (mask.sum() + 1e-8)


# =========================
# GRAPH UTILITIES
# =========================
def make_bidirectional(edge_index):
    """
    Convert directed edges to undirected (bidirectional)
    """
    src, tgt = edge_index
    reversed_edges = np.vstack((tgt, src))
    return np.hstack((edge_index, reversed_edges))


# =========================
# DEBUG HELPERS
# =========================
def check_nan(X, name="Tensor"):
    if np.isnan(X).any():
        print(f"⚠️ Warning: {name} contains NaN values!")


def check_tensor(tensor, name="Tensor"):
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} min: {tensor.min().item():.4f}")
    print(f"{name} max: {tensor.max().item():.4f}")