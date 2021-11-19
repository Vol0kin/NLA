import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import kktsolvers.inequality as kkt_ineq
import kktsolvers.general as kkt_general

def load_problem(problem_dir, n):
    # Compute other dimensions
    p = n // 2
    m = n * 2

    # Load data
    A_dad = np.loadtxt(f'{problem_dir}/A.dad')
    C_dad = np.loadtxt(f'{problem_dir}/C.dad')
    G_dad = np.loadtxt(f'{problem_dir}/G.dad')
    b_dad = np.loadtxt(f'{problem_dir}/b.dad')
    d_dad = np.loadtxt(f'{problem_dir}/d.dad')
    g_dad = np.loadtxt(f'{problem_dir}/g.dad')

    # Transform data to numpy arrays
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

    # Set values for initial point
    x_0 = np.zeros(n)
    s_0, gamma_0, lamb_0 = np.ones(m), np.ones(p), np.ones(m)

    return G, A, C, g, b, d, x_0, gamma_0, lamb_0, s_0


def compute_value(G, g, x):
    return 0.5 * np.dot(x.T, np.dot(G, x)) + np.dot(g.T, x)


def squared_error(g, x):
    return np.linalg.norm(-g - x)


n_iters = []
times = []
condition_numbers = []
errors = []
problem_sizes = [10, 25, 50, 100, 200, 500, 1000]

# Set random number generator seed
np.random.seed(42)

print('################################ KKT INEQUALITY ################################')

for n in problem_sizes:
    # Define initial conditions for every problem
    m = 2 * n

    G = np.eye(n)
    C = np.hstack((np.eye(n), -np.eye(n)))
    d = -10 * np.ones(m)
    g = np.random.normal(size=n)
    x_0 = np.zeros(n)
    s_0 = np.ones(m)
    lamb_0 = np.ones(m)

    print(f'Problem size: {n}\n\n')

    # Solve with every method and show solution and time
    x, i, total_t, cond_nums = kkt_ineq.lu_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=False)
    error = squared_error(g, x)
    times.append(total_t)
    n_iters.append(i)
    condition_numbers.append(cond_nums)
    errors.append(error)
    print(f'Linear system solution squared error:\t\t{error}\tNum. iterations: {i}\tTime: {total_t}')

    x, i, total_t, cond_nums = kkt_ineq.ldlt_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=False)
    error = squared_error(g, x)
    times.append(total_t)
    n_iters.append(i)
    condition_numbers.append(cond_nums)
    errors.append(error)
    print(f'LDL^t factorization solution squared error:\t{error}\tNum. iterations: {i}\tTime: {total_t}')

    x, i, total_t, cond_nums = kkt_ineq.cholesky_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=False)
    error = squared_error(g, x)
    times.append(total_t)
    n_iters.append(i)
    condition_numbers.append(cond_nums)
    errors.append(error)
    print(f'Cholesky factorization solution squared error:\t{error}\tNum. iterations: {i}\tTime: {total_t}\n\n')

    print('-'*80)
    print('\n\n')


print('################################ KKT GENERAL CASE ################################')

G, A, C, g, b, d, x_0, gamma_0, lamb_0, s_0 = load_problem('optpr1', 100)

x_sol, i, total_t, condition_numbers = kkt_general.lu_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
f_sol = compute_value(G, g, x_sol)
print(f'Linear system solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

x_sol, i, total_t, condition_numbers = kkt_general.ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
f_sol = compute_value(G, g, x_sol)
print(f'LDL^t factorization solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

G, A, C, g, b, d, x_0, gamma_0, lamb_0, s_0 = load_problem('optpr2', 1000)

x_sol, i, total_t, condition_numbers = kkt_general.lu_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
f_sol = compute_value(G, g, x_sol)
print(f'Linear system solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

x_sol, i, total_t, condition_numbers = kkt_general.ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
f_sol = compute_value(G, g, x_sol)
print(f'LDL^t factorization solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')