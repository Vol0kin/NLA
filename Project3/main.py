import numpy as np
import pandas as pd
import scipy.io
import page_rank

def run_experiment(G, tolerances, power_method, output_file):
    # Define lists containing results of experiments
    eigenvectors = []
    orders = []
    iters = []
    times = []

    for tol in tolerances:
        x, order, iter, time = power_method(G, tol=tol)

        eigenvectors.append(x)
        orders.append(order)
        iters.append(iter)
        times.append(time)

    # Take as best result the one that uses the smallest tolerance
    best_eigenvector = eigenvectors[0]
    best_order = orders[0]

    # For each of the results, compute the error (squared norm), the order
    # difference and the number of incorrectly indexed pages with respect to the
    # best result
    errors = []
    first_diff_page = []
    num_diff_pages = []

    for eig, order in zip(eigenvectors, orders):
        errors.append(np.linalg.norm(best_eigenvector - eig))

        diff_order = np.where(best_order != order)[0]

        num_diff_pages.append(len(diff_order))

        if len(diff_order) > 0:
            first_diff_page.append(diff_order[0])
        else:
            first_diff_page.append(0)
    
    results = {
        'Tolerance': tolerances,
        'Time (s)': times,
        'Num. iterations': iters,
        'Error': errors,
        'Num. incorrect pages': num_diff_pages,
        'First incorrect page': first_diff_page
    }

    results_df = pd.DataFrame(data=results)

    # Show results of experiments
    print(results_df)

    # Save results to csv
    results_df.to_csv(output_file)


# Load matrix
G = scipy.io.mmread('p2p-Gnutella30.mtx')

tolerances = [1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1]

pd.set_option('display.precision', 4)

print('Results for PageRank with regular power method\n')
run_experiment(G, tolerances, page_rank.power_method, 'power_method.csv')

print('\n\nResults for PageRank with modified power method\n')
run_experiment(G, tolerances, page_rank.power_method_no_matrix, 'power_method_no_matrix.csv')
