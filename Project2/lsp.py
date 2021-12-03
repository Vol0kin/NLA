import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

def solve_lsp_svd(A, b, tol=1e-10):
    # Set full_matrices=false so that u is (M, K) and v is (K, N)
    u, s, v = np.linalg.svd(A, full_matrices=False)

    # Compute rank of diagonal matrix (array in this case) containing eigenvalues
    # This is used for the rank deficient case
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
    rk = np.linalg.matrix_rank(A)

    Q, R, p = spla.qr(A, pivoting=True)
    y = np.dot(Q.T, b)

    # Create permutation matrix (row permutation)
    I = np.eye(n)
    P_mat = I[p]

    R1 = R[:rk, :rk]
    y1 = y[:rk]

    x = spla.solve_triangular(R1, y1)

    # Get full solution
    x = np.concatenate((x, np.zeros(n - rk)))

    # Multiply by transpose of permutation matrix, which gives column permutations
    x = np.dot(P_mat.T, x)

    return x


Ab = np.loadtxt('datafile.txt')
a, b = Ab[:, 0], Ab[:, 1]
n_points = len(a)


degrees = [i for i in range(2, 16)]

for deg in degrees:
    A = np.array([[p ** i for i in range(deg + 1)] for p in a])
    x = solve_lsp_qr_full_rank(A, b)

    # plt.scatter(a, b)
    # plt.scatter(a, np.dot(A, x))
    # plt.show()


# Read input file
Ab = np.loadtxt('datafile2.csv', delimiter=',')
A, b = Ab[:, :-1], Ab[:, -1]

x_svd = solve_lsp_svd(A, b)
x_qr = solve_lsp_qr_rank_deficient(A, b)

svd_err = np.linalg.norm(np.dot(A, x_svd) - b, 2)
qr_err = np.linalg.norm(np.dot(A, x_qr) - b, 2)

print(x_svd)
print(svd_err)

print(x_qr)
print(qr_err)
