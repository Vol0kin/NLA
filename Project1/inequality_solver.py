import numpy as np
import scipy as sp
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
        M_kkt[-m:, -m:] = -np.diagflat(s / lamb)

        # Compute LDL^T factorization
        L_kkt, D_kkt, _ = spla.ldl(M_kkt)

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(M_kkt))
            condition_numbers.append(np.linalg.cond(L_kkt))
            condition_numbers.append(np.linalg.cond(D_kkt))
            condition_numbers.append(np.linalg.cond(np.dot(D_kkt, L_kkt.T)))

        # Step 1: Solve system
        rh_vector = np.hstack((r_1, r_2 - r_3 / lamb))

        y = spla.solve_triangular(L_kkt, -rh_vector, lower=True)
        u = y / D_kkt.diagonal()
        d_z = spla.solve_triangular(L_kkt.T, u)
        d_x, d_lamb = d_z[:n], d_z[n:n+m]
        d_s = (-r_3 - s * d_lamb) / lamb

        # Step 2: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 3: Compute correctors
        mu_tilda = np.dot(s + alpha * d_s, lamb + alpha * d_lamb) / m
        sigma = (mu_tilda / mu) ** 3

        # Step 4: Corrector substep
        # The matrix multiplication D_s D_{\lambda} e can be substituted by the
        # element-wise multiplication of d_s and d_{\lambda}
        r_3 = r_3 + d_s * d_lamb - sigma * mu * e
        rh_vector = np.hstack((r_1, r_2 - r_3 / lamb))

        y = spla.solve_triangular(L_kkt, -rh_vector, lower=True)
        u = y / D_kkt.diagonal()
        d_z = spla.solve_triangular(L_kkt.T, u)
        d_x, d_lamb = d_z[:n], d_z[n:n+m]
        d_s = (-r_3 - s * d_lamb) / lamb

        # Step 5: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 6: Update values
        x += FACTOR * alpha * d_x
        lamb += FACTOR * alpha * d_lamb
        s += FACTOR * alpha * d_s

        r_1, r_2, r_3, rh_vector = solve_system(G, C, x, g, d, lamb, s)

        mu = np.dot(s, lamb) / m

    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers


def kkt_inequality_cholesky_solver(G, C, g, d, lamb_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
    # Copy initial values (these will be modified later on)
    x = np.copy(x_0)
    lamb = np.copy(lamb_0)
    s = np.copy(s_0)

    # Get dimensions
    n, m = C.shape

    # Define values used in the algorithm
    e = np.ones(m)
    FACTOR = 0.95
    i = 0
    condition_numbers = []

    start_t = time.time()

    # Compute initial values
    r_1, r_2, r_3, _ = solve_system(G, C, x, g, d, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_1) >= tol and np.linalg.norm(r_2) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute G hat matrix and vector
        diag_inv_s_lamb = np.diagflat(lamb / s)

        G_hat = G + np.dot(C, np.dot(diag_inv_s_lamb, C.T))
        rh_vector = r_1 - np.dot(C, (-r_3 + lamb * r_2) / s)

        # Compute Cholesky factorization (no need to store previous value of G hat)
        G_hat = spla.cholesky(G_hat)

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(G_hat))
            condition_numbers.append(np.linalg.cond(G_hat.T))

        # Step 1: Solve system
        y = spla.solve_triangular(G_hat, -rh_vector, lower=True)
        d_x = spla.solve_triangular(G_hat.T, y)
        d_lamb = (-r_3 + lamb * r_2) / s - np.dot(diag_inv_s_lamb, np.dot(C.T, d_x))
        d_s = -r_2 + np.dot(C.T, d_x)

        # Step 2: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 3: Compute correctors
        mu_tilda = np.dot(s + alpha * d_s, lamb + alpha * d_lamb) / m
        sigma = (mu_tilda / mu) ** 3

        # Step 4: Corrector substep
        # The matrix multiplication D_s D_{\lambda} e can be substituted by the
        # element-wise multiplication of d_s and d_{\lambda}
        r_3 = r_3 + d_s * d_lamb - sigma * mu * e
        rh_vector = r_1 - np.dot(C, (-r_3 + lamb * r_2) / s)

        y = spla.solve_triangular(G_hat, -rh_vector, lower=True)
        d_x = spla.solve_triangular(G_hat.T, y)
        d_lamb = (-r_3 + lamb * r_2) / s - np.dot(diag_inv_s_lamb, np.dot(C.T, d_x))
        d_s = -r_2 + np.dot(C.T, d_x)

        # Step 5: Step-size correction substep
        alpha = Newton_step(lamb, d_lamb, s, d_s)

        # Step 6: Update values
        x += FACTOR * alpha * d_x
        lamb += FACTOR * alpha * d_lamb
        s += FACTOR * alpha * d_s

        r_1, r_2, r_3, _ = solve_system(G, C, x, g, d, lamb, s)

        mu = np.dot(s, lamb) / m

    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers


def solve_full_system(G, A, C, x, g, b, d, gamma, lamb, s):
    r_L = np.dot(G, x) + g - np.dot(A, gamma) - np.dot(C, lamb)
    r_A = b - np.dot(A.T, x)
    r_C = s + d - np.dot(C.T, x)
    r_s = s * lamb

    rh_vector = np.hstack((r_L, r_A, r_C, r_s))

    return r_L, r_A, r_C, r_s, rh_vector


def kkt_equality_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
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
    r_L, r_A, r_C, r_s, rh_vector = solve_full_system(G, A, C, x, g, b, d, gamma, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_L) >= tol and np.linalg.norm(r_A) >= tol and np.linalg.norm(r_C) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(M_kkt))

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

        r_L, r_A, r_C, r_s, rh_vector = solve_full_system(G, A, C, x, g, b, d, gamma, lamb, s)

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
    i += 1

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


def kkt_equality_ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0, max_iter=100, tol=1e-16, cond_num=False):
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
    r_L, r_A, r_C, r_s, _ = solve_full_system(G, A, C, x, g, b, d, gamma, lamb, s)
    mu = np.dot(s, lamb) / m

    while np.linalg.norm(r_L) >= tol and np.linalg.norm(r_A) >= tol and np.linalg.norm(r_C) >= tol and np.linalg.norm(mu) >= tol and i < max_iter:
        i += 1

        # Compute variable part of M_kkt matrix
        M_kkt[-m:, -m:] = -np.diagflat(s / lamb)

        # Compute LDL^T factorization
        L_kkt, D_kkt, perm = spla.ldl(M_kkt)

        # Compute condition number
        if cond_num:
            condition_numbers.append(np.linalg.cond(M_kkt))

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

        r_L, r_A, r_C, r_s, _ = solve_full_system(G, A, C, x, g, b, d, gamma, lamb, s)

        mu = np.dot(s, lamb) / m

    total_t = time.time() - start_t

    return x, i, total_t, condition_numbers


def load_problem(problem_dir, n):
    p = n // 2
    m = n * 2

    A_dad = np.loadtxt(f'{problem_dir}/A.dad' )
    C_dad = np.loadtxt(f'{problem_dir}/C.dad' )
    G_dad = np.loadtxt(f'{problem_dir}/G.dad' )
    b_dad = np.loadtxt(f'{problem_dir}/b.dad' )
    d_dad = np.loadtxt(f'{problem_dir}/d.dad' )
    g_dad = np.loadtxt(f'{problem_dir}/g.dad' )

    # Transform to numpy format
    G = sp.sparse.coo_matrix((G_dad[:, 2], (G_dad[:, 0] - 1, G_dad[:, 1] - 1)), shape=(n, n)).toarray()
    G = G + G.T - np.diag(G.diagonal())

    A = sp.sparse.coo_matrix((A_dad[:, 2], (A_dad[:, 0] - 1, A_dad[:, 1] - 1)), shape=(n, p)).toarray()
    C = sp.sparse.coo_matrix((C_dad[:, 2], (C_dad[:, 0] - 1, C_dad[:, 1] - 1)), shape=(n, m)).toarray()

    b = np.zeros(p)
    g = np.zeros(n)
    d = np.zeros(m)
    b[(b_dad[:,0]-1).astype(int)] = b_dad[:,1]
    g[(g_dad[:,0]-1).astype(int)] = g_dad[:,1]
    d[(d_dad[:,0]-1).astype(int)] = d_dad[:,1]

    # Set z0
    x_0 = np.zeros(n)
    s_0, gamma_0, lamb_0 = np.ones(m), np.ones(p), np.ones(m)

    return G, A, C, g, b, d, x_0, gamma_0, lamb_0, s_0


n = 5
m = 2 * n

G = np.eye(n)
C = np.hstack((np.eye(n), -np.eye(n)))
d = -10 * np.ones(m)
g = np.random.normal(size=n)
x_0 = np.zeros(n)
s_0 = np.ones(m)
lamb_0 = np.ones(m)

print(f'Starting point: {g}')

# Solve with every method and show solution and time
x_sol, i, total_t, condition_numbers = kkt_inequality_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=False)
print(f'Linear system solution: {x_sol}\tNum. iterations: {i}\tTime: {total_t}s')

x_sol, i, total_t, condition_numbers = kkt_inequality_ldlt_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=False)
print(f'LDL^t factorization solution: {x_sol}\tNum. iterations: {i}\tTime: {total_t}s')

x_sol, i, total_t, condition_numbers = kkt_inequality_cholesky_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=False)
print(f'Cholesky factorization solution: {x_sol}\tNum. iterations: {i}\tTime: {total_t}s')

G, A, C, g, b, d, x_0, gamma_0, lamb_0, s_0 = load_problem('optpr1', 100)

x_sol, i, total_t, condition_numbers = kkt_equality_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
f_sol = 0.5 * np.dot(x_sol.T, np.dot(G, x_sol)) + np.dot(g.T, x_sol)
print(f'Linear system solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

x_sol, i, total_t, condition_numbers = kkt_equality_ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
f_sol = 0.5 * np.dot(x_sol.T, np.dot(G, x_sol)) + np.dot(g.T, x_sol)
print(f'LDL^t factorization solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

G, A, C, g, b, d, x_0, gamma_0, lamb_0, s_0 = load_problem('optpr2', 1000)

x_sol, i, total_t, condition_numbers = kkt_equality_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
f_sol = 0.5 * np.dot(x_sol.T, np.dot(G, x_sol)) + np.dot(g.T, x_sol)
print(f'Linear system solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

x_sol, i, total_t, condition_numbers = kkt_equality_ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
f_sol = 0.5 * np.dot(x_sol.T, np.dot(G, x_sol)) + np.dot(g.T, x_sol)
print(f'LDL^t factorization solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')