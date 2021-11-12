import numpy as np
import scipy.linalg as spla
import time
from utils import Newton_step

def solve_system(G, C, x, g, d, lamb, s):
    r_L = np.dot(G, x) + g - np.dot(C, lamb)
    r_C = s + d - np.dot(C.T, x)
    r_s = s * lamb

    rh_vector = np.hstack((r_L, r_C, r_s))

    return r_L, r_C, r_s, rh_vector


def kkt_inequality_solver(G, C, g, d, lamb_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
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

    M_kkt[-m:, n:n+m] = np.diagflat(s)
    M_kkt[-m:, -m:] = np.diagflat(lamb)
    M_kkt[n:n+m, -m:] = np.eye(m)

    # Compute initial values
    r_L, r_C, r_s, rh_vector = solve_system(G, C, x, g, d, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_L) >= tol and np.linalg.norm(r_C) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(M_kkt))

        # Step 1: Solve system
        d_z = np.linalg.solve(M_kkt, -rh_vector)
        d_x, d_lamb, d_s = d_z[:n], d_z[n:n+m], d_z[-m:]

        # Step 2: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 3: Compute correctors
        mu_tilda = np.dot(s + alpha * d_s, lamb + alpha * d_lamb) / m
        sigma = (mu_tilda / mu) ** 3

        # Step 4: Corrector substep
        # The multiplication D_{\sigma} D_{\lambda} e can be substituted by the
        # element-wise multiplication of d_s and d_\lambda
        r_s = r_s + d_s * d_lamb - sigma * mu * e
        rh_vector = np.hstack((r_L, r_C, r_s))

        d_z = np.linalg.solve(M_kkt, -rh_vector)
        d_x, d_lamb, d_s = d_z[:n], d_z[n:n+m], d_z[-m:]

        # Step 5: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 6: Update values and M_kkt matrix
        x += FACTOR * alpha * d_x
        lamb += FACTOR * alpha * d_lamb
        s += FACTOR * alpha * d_s

        M_kkt[-m:, n:n+m] = np.diagflat(s)
        M_kkt[-m:, -m:] = np.diagflat(lamb)

        r_L, r_C, r_s, rh_vector = solve_system(G, C, x, g, d, lamb, s)

        mu = np.dot(s, lamb) / m


    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers



def kkt_inequality_ldlt_solver(G, C, g, d, lamb_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
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
    r_1, r_2, r_3, rh_vector = solve_system(G, C, x, g, d, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_1) >= tol and np.linalg.norm(r_2) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute variable block of the matrix
        inv_diag_lamb = np.diagflat(1 / lamb)
        diag_s = np.diagflat(s)
        M_kkt[-m:, -m:] = -np.dot(inv_diag_lamb, diag_s)

        # Compute LDL^T factorization
        L_kkt, D_kkt, _ = spla.ldl(M_kkt)

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(M_kkt))
            condition_numbers.append(np.linalg.cond(L_kkt))
            condition_numbers.append(np.linalg.cond(D_kkt))
            condition_numbers.append(np.linalg.cond(np.dot(D_kkt, L_kkt.T)))

        # Step 1: Solve system
        rh_vector = np.hstack((r_1, r_2 - np.dot(inv_diag_lamb, r_3)))

        y = spla.solve_triangular(L_kkt, -rh_vector, lower=True)
        d_z = spla.solve_triangular(np.dot(D_kkt, L_kkt.T), y)
        d_x, d_lamb = d_z[:n], d_z[n:n+m]
        d_s = np.dot(inv_diag_lamb, -r_3 - np.dot(diag_s, d_lamb))

        # Step 2: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 3: Compute correctors
        mu_tilda = np.dot(s + alpha * d_s, lamb + alpha * d_lamb) / m
        sigma = (mu_tilda / mu) ** 3

        # Step 4: Corrector substep
        # The multiplication D_{\sigma} D_{\lambda} e can be substituted by the
        # element-wise multiplication of d_s and d_\lambda
        r_3 = r_3 + d_s * d_lamb - sigma * mu * e
        rh_vector = np.hstack((r_1, r_2 - np.dot(inv_diag_lamb, r_3)))

        y = spla.solve_triangular(L_kkt, -rh_vector, lower=True)
        d_z = spla.solve_triangular(np.dot(D_kkt, L_kkt.T), y)
        d_x, d_lamb = d_z[:n], d_z[n:n+m]
        d_s = np.dot(inv_diag_lamb, -r_3 - np.dot(diag_s, d_lamb))

        # Step 5: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 6: Update values and M_kkt matrix
        x += FACTOR * alpha * d_x
        lamb += FACTOR * alpha * d_lamb
        s += FACTOR * alpha * d_s

        r_1, r_2, r_3, rh_vector = solve_system(G, C, x, g, d, lamb, s)

        mu = np.dot(s, lamb) / m

    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers


n = 5
m = 2 * n

G = np.eye(n)
C = np.hstack((np.eye(n), -np.eye(n)))
d = -10 * np.ones(m)
g = np.random.normal(size=n)
x_0 = np.zeros(n)
s_0 = np.ones(m)
lamb_0 = np.ones(m)

x_sol, i, total_t, condition_numbers = kkt_inequality_ldlt_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=False)
print(f'Starting point: {g}')
print(f'Solution: {x_sol}\tNum. iterations: {i}\tTime: {total_t}s')

print('Condition numbers')
print(condition_numbers)
