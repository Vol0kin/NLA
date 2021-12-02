import numpy as np

# Read input file
A = np.loadtxt('datafile.txt')

# print(A)

Ab = np.loadtxt('datafile2.csv', delimiter=',')
A, b = Ab[:, :-1], Ab[:, -1]

u, s, v = np.linalg.svd(A, full_matrices=False)
rank = np.sum(s > 1.e-10)

print('rank: ', rank)


# print(A)

print('u: ', u.shape)
print('s: ', s.shape)
print('v: ', v.shape)

u = u[:, :rank]
s = s[:rank]
v = v[:rank, :]


s_inv = 1 / s

S = np.diagflat(s_inv)
print(S)
# print(u.shape, S.shape, v.shape)

SVD = np.dot(v.T, np.dot(S, u.T))
pinv = np.linalg.pinv(A)

print(SVD == pinv)


print('SVD: ', SVD.shape, SVD)
print('Pinv: ', pinv.shape, pinv)

x = np.dot(SVD, b)
print(x)

x = np.dot(pinv, b)
print(x)

print(np.dot(A, x), '\n', b)
