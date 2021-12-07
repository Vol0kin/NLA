import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(X, covariance=True, main_axis=1):
    # Transpose data so that rows contain the variables and columns the samples
    if main_axis == 0:
        X = X.T
    
    # Center and transform data accordingly
    if covariance:
        # Center the data according to the mean (covariance matrix)
        X = X - np.mean(X, axis=1).reshape(-1, 1)
    else:
        # Center and standarize the data (correlation matrix)
        X = (X - np.mean(X, axis=1).reshape(-1, 1)) / np.std(X, axis=1).reshape(-1, 1)

    n = X.shape[1]
    Y = X.T / np.sqrt(n - 1)

    # Compute SVD of Y
    _, s, v = np.linalg.svd(Y, full_matrices=False)
    r = np.sum(s > 1e-10)

    principal_components = np.dot(v[:r, :], X)
    variance_proportion = s[:r] / np.sum(s[:r] ** 2)
    acc_variance = np.cumsum(variance_proportion)
    std = s[:r]

    return principal_components, variance_proportion, acc_variance, std


##################### Problem 1 #####################
# Load data
X = np.loadtxt('example.dat')

# Perform PCA using covariance matrix
principal_components, variance_proportion, acc_variance, std = PCA(X, main_axis=0)

# Perform PCA using correlation matrix
principal_components, variance_proportion, acc_variance, std = PCA(X, covariance=False, main_axis=0)


##################### Problem 2 #####################
