import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

def solve_lsp_svd(A, b, tol=None):
    # Set full_matrices=False so that u is (M, K) and v is (K, N)
    u, s, v = np.linalg.svd(A, full_matrices=False)

    # Compute rank of diagonal matrix (array in this case) containing singular values.
    # This is used for the rank deficient case. The formula for the tolerance
    # has been extracted from Numpy's documentation.
    # Reference: https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html
    if tol is None:
        tol = s[0] * max(A.shape) * np.finfo(float).eps

    rank = np.sum(s > tol)

    u, s, v = u[:, :rank], s[:rank], v[:rank, :]

    # Compute Moore-Penrose inverse using elements from SVD
    A_pinv = np.dot(v.T, np.dot(np.diagflat(1 / s), u.T))

    # Compute solution
    x = np.dot(A_pinv, b)

    return x


def solve_lsp_qr_full_rank(A, b):
    # Get number of independent columns for the thin decomposition
    n = A.shape[1]

    Q, R = spla.qr(A)
    y = np.dot(Q.T, b)

    R1 = R[:n, :n]
    y1 = y[:n]

    x = spla.solve_triangular(R1, y1)

    return x


def solve_lsp_qr_rank_deficient(A, b):
    n = A.shape[1]
    rank = np.linalg.matrix_rank(A)

    Q, R, p = spla.qr(A, pivoting=True)
    y = np.dot(Q.T, b)

    # Create permutation matrix (row permutation)
    I = np.eye(n)
    P_mat = I[p]

    R1 = R[:rank, :rank]
    y1 = y[:rank]

    x = spla.solve_triangular(R1, y1)

    # Get full solution
    x = np.concatenate((x, np.zeros(n - rank)))

    # Multiply by transpose of permutation matrix, which gives column permutations
    x = np.dot(P_mat.T, x)

    return x


def compute_error(A, b, x):
    return np.linalg.norm(np.dot(A, x) - b, 2)


def plot_error(degrees, errors):
    plt.plot(degrees, errors, 'bo-')

    plt.xlabel('Degree of the polynomial')
    plt.ylabel('Error')

    plt.show()


##################### Problem 1 #####################
# Read input file and load data
data = np.loadtxt('datafile.txt')
a, b = data[:, 0], data[:, 1]
n_points = len(a)

degrees = [i for i in range(2, 19)]

svd_errors = []
svd_errors_fix_tol = []
qr_errors = []

print('##################### Problem 1 #####################\n\n')

for deg in degrees:
    A = np.array([[p ** i for i in range(deg + 1)] for p in a])
    x_svd = solve_lsp_svd(A, b)
    x_svd_fix_tol = solve_lsp_svd(A, b, tol=1e-10)
    x_qr = solve_lsp_qr_full_rank(A, b)

    svd_error = compute_error(A, b, x_svd)
    svd_error_fix_tol = compute_error(A, b, x_svd_fix_tol)
    qr_error = compute_error(A, b, x_qr)

    svd_errors.append(svd_error)
    svd_errors_fix_tol.append(svd_error_fix_tol)
    qr_errors.append(qr_error)

    print(f'\nPolynomial degree: {deg}')
    print(f'Errors:\tSVD: {svd_error}\tSVD fixed tolerance:{svd_error_fix_tol}\tQR: {qr_error}')

plot_error(degrees, svd_errors)
plot_error(degrees, svd_errors_fix_tol)
plot_error(degrees, qr_errors)

##################### Problem 2 #####################
# Read input file and load data
data = np.loadtxt('datafile2.csv', delimiter=',')
A, b = data[:, :-1], data[:, -1]

x_svd = solve_lsp_svd(A, b)
x_qr = solve_lsp_qr_rank_deficient(A, b)

print('\n\n##################### Problem 2 #####################\n\n')

svd_error = compute_error(A, b, x_svd)
qr_error = compute_error(A, b, x_qr)

print(f'Error using SVD: {svd_error}')
print(f'Error using QR: {qr_error}')
