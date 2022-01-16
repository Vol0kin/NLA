import numpy as np
import time

def power_method(G, tol=1e-4, m=0.15):
    start_t = time.time()

    # Compute out-degree for each page
    n = np.sum(G, axis=0)
    n = np.squeeze(np.asarray(n))

    num_links = len(n)

    # Since G is a sparse matrix, the data can be represented as 3 arrays
    # containing the rows containing values, the columns containing values and
    # the actual values
    in_links = G.row
    out_links = G.col

    # The G * D matrix product is going to produce a sparse matrix. The best
    # way to calculate it is to iterate over the columns that contain values
    # (outgoing links) and to compute the value of 1 / n_j
    values = np.array([1 / n[out] for out in out_links])
    values = (1 - m) * values

    # Initialize array containing PR scores
    x = np.ones(num_links) / num_links

    z = np.ones(num_links) / num_links
    z[n != 0] *= m

    norm_diff = 1
    i = 0

    while norm_diff > tol:
        x_prev = x
        x = np.zeros(num_links)

        # in_links is used to reference the output position of the sparse dot
        # product between the matrix and the PR vector in the current iteration.
        # out_links is used to acces the particular values of the PR vector that
        # take part in the sparse dot product.
        for k in range(len(values)):
            x[in_links[k]] += values[k] * x_prev[out_links[k]]

        x += np.dot(z, x_prev)
        norm_diff = np.linalg.norm(x - x_prev, ord=np.inf)

        i += 1

    order = np.argsort(x)[::-1]
    total_t = time.time() - start_t

    return x, order, i, total_t


def power_method_no_matrix(G, tol=1e-4, m=0.15):
    start_t = time.time()

    in_links = G.row
    out_links = G.col

    num_links = G.shape[0]

    # Links will contain the pages linked from page_j
    links = [[] for _ in range(num_links)]
    n = np.zeros(num_links)

    for j in range(len(out_links)):
        links[out_links[j]].append(in_links[j])
        n[out_links[j]] += 1

    x = np.ones(num_links) / num_links
    i = 0
    norm_diff = 1

    while norm_diff > tol:
        x_prev = x
        x = np.zeros(num_links)

        for j in range(num_links):
            if (n[j] == 0):
                # If page_j doesn't link to any page, x_j / n is added to the whole
                # vector
                x += x_prev[j] / num_links
            else:
                # If it does, update scores of linked pages by x_j / n_j
                x[links[j]] += x_prev[j] / n[j]

        x = (1 - m)  * x + m / num_links
        norm_diff = np.linalg.norm(x - x_prev, ord=np.inf)

        i += 1

    order = np.argsort(x)[::-1]
    total_t = time.time() - start_t

    return x, order, i, total_t
