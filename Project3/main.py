import scipy.io
import page_rank

G = scipy.io.mmread('p2p-Gnutella30.mtx')
x, order, i, time = page_rank.power_method(G)

print(order)
print(x[order])