import numpy as np

def normalize_data(X):
    """Normalize the data using min-max scaling"""
    return (X - X.min()) / (X.max() - X.min())

def train_test_split(X, y, test_size=0.2):
    """Split data into training and test sets"""
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]