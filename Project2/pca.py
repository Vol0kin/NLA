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

    tol = np.finfo(float).eps
    rank = np.sum(s > tol)

    s, v = s[:rank], v[:rank, :]

    principal_components = np.dot(v, X)
    variance_proportion = s**2 / np.sum(s**2)
    acc_variance = np.cumsum(variance_proportion) * 100
    std = s

    return principal_components, variance_proportion, acc_variance, std


def plot_pca(principal_components):
    plt.scatter(principal_components[0], principal_components[1])

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.show()


def kaiser_rule(eigenvalues):
    x = np.arange(1, len(eigenvalues) + 1)

    plt.plot(x, eigenvalues, 'bo-')
    plt.axhline(1, color='r', linestyle='-.')

    plt.xlabel('Number of PC')
    plt.ylabel('Eigenvalues')
    plt.xticks(x)

    plt.show()


def variance_rule(acc_variance):
    x = np.arange(1, len(acc_variance) + 1)

    plt.plot(x, acc_variance, 'bo-')
    plt.axhline(75, color='r', linestyle='-.')

    plt.xlabel('Number of PC')
    plt.ylabel('Cummulative percentage of explained variance')
    plt.xticks(x)

    plt.show()


##################### Problem 1 #####################
# Load data
X = np.loadtxt('example.dat')

# Perform PCA using covariance matrix
principal_components, variance_proportion, acc_variance, std = PCA(X, main_axis=0)

plot_pca(principal_components)
kaiser_rule(std**2)
variance_rule(acc_variance)

# Perform PCA using correlation matrix
principal_components, variance_proportion, acc_variance, std = PCA(X, covariance=False, main_axis=0)

plot_pca(principal_components)
kaiser_rule(std**2)
variance_rule(acc_variance)


##################### Problem 2 #####################
df = pd.read_csv('RCsGoff.csv')

# Remove 'gene' column
df = df.drop('gene', axis=1)
experiments = df.columns.tolist()
X = df.values

principal_components, variance_proportion, acc_variance, std = PCA(X)

plot_pca(principal_components)
kaiser_rule(std**2)
variance_rule(acc_variance)
