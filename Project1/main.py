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
lu_cond_nums = []
ldl_cond_nums = []
cholesky_cond_nums = []
errors = []
problem_sizes = [10, 25, 50, 100, 200, 500, 1000]

# Set random number generator seed
np.random.seed(7)

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
    x, i, total_t, _ = kkt_ineq.lu_solver(G, C, g, d, lamb_0, s_0, x_0)
    _, _, _ , cond_nums = kkt_ineq.lu_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=True)
    error = squared_error(g, x)
    times.append(total_t)
    n_iters.append(i)
    lu_cond_nums.append(cond_nums)
    errors.append(error)
    print(f'Linear system solution squared error:\t\t{error}\tNum. iterations: {i}\tTime: {total_t}')

    x, i, total_t, _ = kkt_ineq.ldlt_solver(G, C, g, d, lamb_0, s_0, x_0)
    _, _, _, cond_nums = kkt_ineq.ldlt_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=True)
    error = squared_error(g, x)
    times.append(total_t)
    n_iters.append(i)
    ldl_cond_nums.append(cond_nums)
    errors.append(error)
    print(f'LDL^t factorization solution squared error:\t{error}\tNum. iterations: {i}\tTime: {total_t}')

    x, i, total_t, _ = kkt_ineq.cholesky_solver(G, C, g, d, lamb_0, s_0, x_0)
    _, _, _, cond_nums = kkt_ineq.cholesky_solver(G, C, g, d, lamb_0, s_0, x_0, cond_num=True)
    error = squared_error(g, x)
    times.append(total_t)
    n_iters.append(i)
    cholesky_cond_nums.append(cond_nums)
    errors.append(error)
    print(f'Cholesky factorization solution squared error:\t{error}\tNum. iterations: {i}\tTime: {total_t}\n\n')


n_iters = np.array(n_iters).reshape(-1, 3)
times = np.array(times).reshape(-1, 3)
errors = np.array(errors).reshape(-1, 3)

max_cond_lu = [max(lu_cond_nums[i]) for i in range(len(lu_cond_nums))]
max_cond_ldl = [max(ldl_cond_nums[i]) for i in range(len(ldl_cond_nums))]
max_cond_cholesky = [max(cholesky_cond_nums[i]) for i in range(len(cholesky_cond_nums))]

# Plot time results
plt.plot(problem_sizes, times[:, 0], label='LU')
plt.plot(problem_sizes, times[:, 1], label=r'$LDL^T$')
plt.plot(problem_sizes, times[:, 2], label='Cholesky')

plt.xlabel('Problem size (n)')
plt.ylabel('Time (s)')
plt.legend()
plt.show()

# Plot max condition numbers
plt.clf()
plt.plot(problem_sizes, max_cond_lu, label='LU')
plt.plot(problem_sizes, max_cond_ldl, label='LDL')
plt.plot(problem_sizes, max_cond_cholesky, label='Chokesky')

plt.xlabel('Problem size (n)')
plt.ylabel('Max. condition number')
plt.legend()
plt.show()

plt.clf()
plt.plot(problem_sizes, max_cond_lu, label='LU')
plt.plot(problem_sizes, max_cond_cholesky, label='Chokesky')

plt.xlabel('Problem size (n)')
plt.ylabel('Max. condition number')
plt.legend()
plt.show()

# Plot condition numbers per iterations
plt.clf()

for i in range(len(lu_cond_nums)):
    plt.plot(np.arange(1, n_iters[i, 0] + 1), lu_cond_nums[i], label=f'n={problem_sizes[i]}')

plt.xlabel('Num. iterations')
plt.ylabel('Condition number')
plt.legend()
plt.show()


plt.clf()

for i in range(len(lu_cond_nums)):
    plt.plot(np.arange(1, n_iters[i, 1] + 1), ldl_cond_nums[i], label=f'n={problem_sizes[i]}')

plt.xlabel('Num. iterations')
plt.ylabel('Condition number')
plt.legend()
plt.show()


plt.clf()

for i in range(len(lu_cond_nums)):
    plt.plot(np.arange(1, n_iters[i, 2] + 1), cholesky_cond_nums[i], label=f'n={problem_sizes[i]}')

plt.xlabel('Num. iterations')
plt.ylabel('Condition number')
plt.legend()
plt.show()



print('################################ KKT GENERAL CASE ################################')

problem_sizes = [100, 1000]
lu_cond_nums = []
ldl_cond_nums = []
n_iters_lu = []
n_iters_ldl = []

G, A, C, g, b, d, x_0, gamma_0, lamb_0, s_0 = load_problem('optpr1', 100)

x_sol, i, total_t, _ = kkt_general.lu_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
_, _, _, condition_numbers = kkt_general.lu_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0, cond_num=True)
lu_cond_nums.append(condition_numbers)
n_iters_lu.append(i)
f_sol = compute_value(G, g, x_sol)
print(f'Linear system solution:\t{f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

x_sol, i, total_t, _ = kkt_general.ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
_, _, _, condition_numbers = kkt_general.ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0, cond_num=True)
ldl_cond_nums.append(condition_numbers)
n_iters_ldl.append(i)
f_sol = compute_value(G, g, x_sol)
print(f'LDL^t factorization solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

G, A, C, g, b, d, x_0, gamma_0, lamb_0, s_0 = load_problem('optpr2', 1000)

x_sol, i, total_t, _ = kkt_general.lu_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
_, _, _, condition_numbers = kkt_general.lu_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0, cond_num=True)
lu_cond_nums.append(condition_numbers)
n_iters_lu.append(i)
f_sol = compute_value(G, g, x_sol)
print(f'Linear system solution:\t{f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

x_sol, i, total_t, _ = kkt_general.ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0)
_, _, _, condition_numbers = kkt_general.ldlt_solver(G, A, C, g, b, d, lamb_0, gamma_0, s_0, x_0, cond_num=True)
ldl_cond_nums.append(condition_numbers)
n_iters_ldl.append(i)
f_sol = compute_value(G, g, x_sol)
print(f'LDL^t factorization solution: {f_sol}\tNum. iterations: {i}\tTime: {total_t}s')

# Plot condition numbers per iterations
plt.clf()

for i in range(len(lu_cond_nums)):
    plt.plot(np.arange(1, n_iters_lu[i] + 1), lu_cond_nums[i], label=f'n={problem_sizes[i]}')

plt.xlabel('Num. iterations')
plt.ylabel('Condition number')
plt.legend()
plt.show()


plt.clf()

for i in range(len(lu_cond_nums)):
    plt.plot(np.arange(1, n_iters_ldl[i] + 1), ldl_cond_nums[i], label=f'n={problem_sizes[i]}')

plt.xlabel('Num. iterations')
plt.ylabel('Condition number')
plt.legend()
plt.show()
