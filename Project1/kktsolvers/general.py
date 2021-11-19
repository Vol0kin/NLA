import numpy as np
import scipy.linalg as spla
import time
from .utils import Newton_step

def solve_system(G, A, C, x, g, b, d, gamma, lamb, s):
    r_L = np.dot(G, x) + g - np.dot(A, gamma) - np.dot(C, lamb)
    r_A = b - np.dot(A.T, x)
    r_C = s + d - np.dot(C.T, x)
    r_s = s * lamb

    rh_vector = np.hstack((r_L, r_A, r_C, r_s))

    return r_L, r_A, r_C, r_s, rh_vector


def lu_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
    # Copy initial values (these will be modified later on)
    x = np.copy(x_0)
    gamma = np.copy(gamma_0)
    lamb = np.copy(lamb_0)
    s = np.copy(s_0)

    # Get dimensions
    n, m = C.shape
    p = A.shape[1]
    N = n + p + 2 * m

    # Define values used in the algorithm
    e = np.ones(m)
    FACTOR = 0.95
    i = 0
    condition_numbers = []

    start_t = time.time()

    # Create M_kkt matrix
    M_kkt = np.zeros((N, N))

    M_kkt[:n, :n] = G
    M_kkt[:n, n:n+p] = -A
    M_kkt[:n, n+p:-m] = -C
    M_kkt[n:n+p, :n] = -A.T
    M_kkt[n+p:-m, :n] = -C.T

    M_kkt[-m:, -2*m:-m] = np.diagflat(s)
    M_kkt[-m:, -m:] = np.diagflat(lamb)
    M_kkt[-2*m:-m, -m:] = np.eye(m)

    # Compute initial values
    r_L, r_A, r_C, r_s, rh_vector = solve_system(G, A, C, x, g, b, d, gamma, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_L) >= tol and np.linalg.norm(r_A) >= tol and np.linalg.norm(r_C) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(M_kkt, 2))

        # Step 1: Solve system
        d_z = np.linalg.solve(M_kkt, -rh_vector)
        d_x, d_gamma, d_lamb, d_s = d_z[:n], d_z[n:n+p], d_z[-2*m:-m], d_z[-m:]

        # Step 2: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 3: Compute correctors
        mu_tilda = np.dot(s + alpha * d_s, lamb + alpha * d_lamb) / m
        sigma = (mu_tilda / mu) ** 3

        # Step 4: Corrector substep
        # The matrix multiplication D_s D_{\lambda} e can be substituted by the
        # element-wise multiplication of d_s and d_{\lambda}
        r_s = r_s + d_s * d_lamb - sigma * mu * e
        rh_vector = np.hstack((r_L, r_A, r_C, r_s))

        d_z = np.linalg.solve(M_kkt, -rh_vector)
        d_x, d_gamma, d_lamb, d_s = d_z[:n], d_z[n:n+p], d_z[-2*m:-m], d_z[-m:]

        # Step 5: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 6: Update values
        x += FACTOR * alpha * d_x
        gamma += FACTOR * alpha * d_gamma
        lamb += FACTOR * alpha * d_lamb
        s += FACTOR * alpha * d_s

        M_kkt[-m:, -2*m:-m] = np.diagflat(s)
        M_kkt[-m:, -m:] = np.diagflat(lamb)

        r_L, r_A, r_C, r_s, rh_vector = solve_system(G, A, C, x, g, b, d, gamma, lamb, s)

        mu = np.dot(s, lamb) / m

    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers


def solve_block_diagonal(D, y, tol=1e-16):
    n = D.shape[0]
    i = 0
    x = np.zeros_like(y)

    while i < n - 1:
        if max(abs(D[i, i+1]), abs(D[i+1, i])) < tol:
            x[i] = y[i] / D[i, i]
            i += 1
        else:
            x[i:i+2] = np.linalg.solve(D[i:i+2, i:i+2], y[i:i+2])
            i += 2

    x[i] = y[i] / D[i, i]

    return x


def ldl_block_solver(L, D, perm, rh_vector):
    n = L.shape[0]

    # Create inverse permutation array
    inv_perm = np.zeros_like(perm)
    for i in range(n):
        inv_perm[perm[i]] = i

    # Permute L matrix and array
    L = L[perm]
    rh_vector = rh_vector[perm]

    # Solve system
    x = spla.solve_triangular(L, rh_vector, lower=True)
    y = solve_block_diagonal(D, x)
    z = spla.solve_triangular(L.T, y)

    # Permute solution
    z = z[inv_perm]

    return z


def ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
    # Copy initial values (these will be modified later on)
    x = np.copy(x_0)
    gamma = np.copy(gamma_0)
    lamb = np.copy(lamb_0)
    s = np.copy(s_0)

    # Get dimensions
    n, m = C.shape
    p = A.shape[1]
    N = n + p + m

    # Define values used in the algorithm
    e = np.ones(m)
    FACTOR = 0.95
    i = 0
    condition_numbers = []

    start_t = time.time()

    # Create M_kkt matrix
    M_kkt = np.zeros((N, N))

    M_kkt[:n, :n] = G
    M_kkt[:n, n:-m] = -A
    M_kkt[:n, -m:] = -C
    M_kkt[n:-m, :n] = -A.T
    M_kkt[-m:, :n] = -C.T

    # Compute initial values
    r_L, r_A, r_C, r_s, _ = solve_system(G, A, C, x, g, b, d, gamma, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_L) >= tol and np.linalg.norm(r_A) >= tol and np.linalg.norm(r_C) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute variable part of M_kkt matrix
        M_kkt[-m:, -m:] = -np.diagflat(s / lamb)

        # Compute LDL^T factorization
        L_kkt, D_kkt, perm = spla.ldl(M_kkt)

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(M_kkt, 2))

        # Step 1: Solve system
        rh_vector = np.hstack((r_L, r_A, r_C - r_s / lamb))

        d_z = ldl_block_solver(L_kkt, D_kkt, perm, -rh_vector)
        d_x, d_gamma, d_lamb = d_z[:n], d_z[n:-m], d_z[-m:]
        d_s = -(r_s + s * d_lamb) / lamb

        # Step 2: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 3: Compute correctors
        mu_tilda = np.dot(s + alpha * d_s, lamb + alpha * d_lamb) / m
        sigma = (mu_tilda / mu) ** 3

        # Step 4: Corrector substep
        # The matrix multiplication D_s D_{\lambda} e can be substituted by the
        # element-wise multiplication of d_s and d_{\lambda}
        r_s = r_s + d_s * d_lamb - sigma * mu * e
        rh_vector = np.hstack((r_L, r_A, r_C - r_s / lamb))

        d_z = ldl_block_solver(L_kkt, D_kkt, perm, -rh_vector)
        d_x, d_gamma, d_lamb = d_z[:n], d_z[n:-m], d_z[-m:]
        d_s = -(r_s + s * d_lamb) / lamb

        # Step 5: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 6: Update values
        x += FACTOR * alpha * d_x
        gamma += FACTOR * alpha * d_gamma
        lamb += FACTOR * alpha * d_lamb
        s += FACTOR * alpha * d_s

        r_L, r_A, r_C, r_s, _ = solve_system(G, A, C, x, g, b, d, gamma, lamb, s)

        mu = np.dot(s, lamb) / m

    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers
