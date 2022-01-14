import scipy.io
import page_rank

G = scipy.io.mmread('p2p-Gnutella30.mtx')
page_rank.power_method(G)
page_rank.page_rank(G)