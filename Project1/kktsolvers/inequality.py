import numpy as np
import scipy.linalg as spla
import time
from .utils import Newton_step

def solve_system(G, C, x, g, d, lamb, s):
    r_L = np.dot(G, x) + g - np.dot(C, lamb)
    r_C = s + d - np.dot(C.T, x)
    r_s = s * lamb

    rh_vector = np.hstack((r_L, r_C, r_s))

    return r_L, r_C, r_s, rh_vector


def lu_solver(G, C, g, d, lamb_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
    # Copy initial values (these will be modified later on)
    x = np.copy(x_0)
    lamb = np.copy(lamb_0)
    s = np.copy(s_0)

    # Get dimensions
    n, m = C.shape
    N = n + 2 * m

    # Define values used in the algorithm
    e = np.ones(m)
    FACTOR = 0.95
    i = 0
    condition_numbers = []

    start_t = time.time()

    # Create M_kkt matrix
    M_kkt = np.zeros((N, N))

    M_kkt[:n, :n] = G
    M_kkt[n:n+m, :n] = -C.T
    M_kkt[:n, n:n+m] = -C

    M_kkt[n:n+m, -m:] = np.eye(m)

    # Compute initial values
    r_L, r_C, r_s, rh_vector = solve_system(G, C, x, g, d, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_L) >= tol and np.linalg.norm(r_C) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute variable blocks of the matrix
        M_kkt[-m:, n:n+m] = np.diagflat(s)
        M_kkt[-m:, -m:] = np.diagflat(lamb)

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(M_kkt, np.inf))

        # Step 1: Solve system
        d_z = np.linalg.solve(M_kkt, -rh_vector)
        d_x, d_lamb, d_s = d_z[:n], d_z[n:n+m], d_z[-m:]

        # Step 2: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 3: Compute correctors
        mu_tilda = np.dot(s + alpha * d_s, lamb + alpha * d_lamb) / m
        sigma = (mu_tilda / mu) ** 3

        # Step 4: Corrector substep
        # The matrix multiplication D_s D_{\lambda} e can be substituted by the
        # element-wise multiplication of d_s and d_{\lambda}
        r_s = r_s + d_s * d_lamb - sigma * mu * e
        rh_vector = np.hstack((r_L, r_C, r_s))

        d_z = np.linalg.solve(M_kkt, -rh_vector)
        d_x, d_lamb, d_s = d_z[:n], d_z[n:n+m], d_z[-m:]

        # Step 5: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 6: Update values
        x += FACTOR * alpha * d_x
        lamb += FACTOR * alpha * d_lamb
        s += FACTOR * alpha * d_s        

        r_L, r_C, r_s, rh_vector = solve_system(G, C, x, g, d, lamb, s)

        mu = np.dot(s, lamb) / m

    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers


def ldlt_solver(G, C, g, d, lamb_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
    # Copy initial values (these will be modified later on)
    x = np.copy(x_0)
    lamb = np.copy(lamb_0)
    s = np.copy(s_0)

    # Get dimensions
    n, m = C.shape
    N = n + m

    # Define values used in the algorithm
    e = np.ones(m)
    FACTOR = 0.95
    i = 0
    condition_numbers = []

    start_t = time.time()

    # Create M_kkt matrix
    M_kkt = np.zeros((N, N))

    M_kkt[:n, :n] = G
    M_kkt[n:, :n] = -C.T
    M_kkt[:n, n:] = -C

    # Compute initial values
    r_L, r_C, r_s, rh_vector = solve_system(G, C, x, g, d, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_L) >= tol and np.linalg.norm(r_C) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute variable block of the matrix
        M_kkt[-m:, -m:] = -np.diagflat(s / lamb)

        # Compute LDL^T factorization
        L_kkt, D_kkt, _ = spla.ldl(M_kkt)

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(M_kkt, np.inf))

        # Step 1: Solve system
        rh_vector = np.hstack((r_L, r_C - r_s / lamb))

        y = spla.solve_triangular(L_kkt, -rh_vector, lower=True)
        u = y / D_kkt.diagonal()
        d_z = spla.solve_triangular(L_kkt.T, u)
        d_x, d_lamb = d_z[:n], d_z[n:n+m]
        d_s = (-r_s - s * d_lamb) / lamb

        # Step 2: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 3: Compute correctors
        mu_tilda = np.dot(s + alpha * d_s, lamb + alpha * d_lamb) / m
        sigma = (mu_tilda / mu) ** 3

        # Step 4: Corrector substep
        # The matrix multiplication D_s D_{\lambda} e can be substituted by the
        # element-wise multiplication of d_s and d_{\lambda}
        r_s = r_s + d_s * d_lamb - sigma * mu * e
        rh_vector = np.hstack((r_L, r_C - r_s / lamb))

        y = spla.solve_triangular(L_kkt, -rh_vector, lower=True)
        u = y / D_kkt.diagonal()
        d_z = spla.solve_triangular(L_kkt.T, u)
        d_x, d_lamb = d_z[:n], d_z[n:n+m]
        d_s = (-r_s - s * d_lamb) / lamb

        # Step 5: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 6: Update values
        x += FACTOR * alpha * d_x
        lamb += FACTOR * alpha * d_lamb
        s += FACTOR * alpha * d_s

        r_L, r_C, r_s, rh_vector = solve_system(G, C, x, g, d, lamb, s)

        mu = np.dot(s, lamb) / m

    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers


def cholesky_solver(G, C, g, d, lamb_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
    # Copy initial values (these will be modified later on)
    x = np.copy(x_0)
    lamb = np.copy(lamb_0)
    s = np.copy(s_0)

    # Get dimensions
    _, m = C.shape

    # Define values used in the algorithm
    e = np.ones(m)
    FACTOR = 0.95
    i = 0
    condition_numbers = []

    start_t = time.time()

    # Compute initial values
    r_L, r_C, r_s, _ = solve_system(G, C, x, g, d, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_L) >= tol and np.linalg.norm(r_C) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute G hat matrix and vector
        diag_inv_s_lamb = np.diagflat(lamb / s)

        G_hat = G + np.dot(C, np.dot(diag_inv_s_lamb, C.T))
        rh_vector = r_L - np.dot(C, (-r_s + lamb * r_C) / s)

        # Compute Cholesky factorization (no need to store previous value of G hat)
        G_hat = spla.cholesky(G_hat)

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(G_hat, np.inf))

        # Step 1: Solve system
        y = spla.solve_triangular(G_hat, -rh_vector, lower=True)
        d_x = spla.solve_triangular(G_hat.T, y)
        d_lamb = (-r_s + lamb * r_C) / s - np.dot(diag_inv_s_lamb, np.dot(C.T, d_x))
        d_s = -r_C + np.dot(C.T, d_x)

        # Step 2: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 3: Compute correctors
        mu_tilda = np.dot(s + alpha * d_s, lamb + alpha * d_lamb) / m
        sigma = (mu_tilda / mu) ** 3

        # Step 4: Corrector substep
        # The matrix multiplication D_s D_{\lambda} e can be substituted by the
        # element-wise multiplication of d_s and d_{\lambda}
        r_s = r_s + d_s * d_lamb - sigma * mu * e
        rh_vector = r_L - np.dot(C, (-r_s + lamb * r_C) / s)

        y = spla.solve_triangular(G_hat, -rh_vector, lower=True)
        d_x = spla.solve_triangular(G_hat.T, y)
        d_lamb = (-r_s + lamb * r_C) / s - np.dot(diag_inv_s_lamb, np.dot(C.T, d_x))
        d_s = -r_C + np.dot(C.T, d_x)

        # Step 5: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 6: Update values
        x += FACTOR * alpha * d_x
        lamb += FACTOR * alpha * d_lamb
        s += FACTOR * alpha * d_s

        r_L, r_C, r_s, _ = solve_system(G, C, x, g, d, lamb, s)

        mu = np.dot(s, lamb) / m

    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers
