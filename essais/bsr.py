import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg

# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3 ,4, 5, 6])
# A = sparse.bsr_matrix((data, (row, col)), shape=(3, 3), blocksize=(1,1))
# print("A=\n", A.toarray())

# indptr = np.array([0, 1])
# indices = np.array([0])
# data = np.array([[ [1, 0], [2, 1]]])
# A = sparse.bsr_matrix((data, indices, indptr), shape=(2, 2))
# print("A=\n", A.toarray())
# b = np.array( [2,1])
# x = splinalg.spsolve(A.tocsr(), b)
# print("x=\n", x)
# print("b=\n", b)
#
# row = np.array([0, 0, 1, 1])
# col = np.array([0, 1, 0, 1])
# data = np.array([1, 0, 3 ,4])
# Acoo = sparse.coo_matrix((data, (row, col)), shape=(2, 2))
# print("A=\n", Acoo.tocsr().todense())
# print("A=\n", Acoo.tobsr().todense())
# print("A.clo",Acoo.tobsr().blocksize)
# print("A.shape", A.shape)
# print("A", np.ravel(A))
# print("A", A.ravel())

A = np.array([[ [1,2], [3,4] ],[ [5,6], [7,8] ],[ [9,10], [11,12] ] ])
B = []
for i in range(2):
    B.append([])
    for j in range(2):
        B[i].append(np.array([ 1 + 2*i + j + 4*k for k in range(3)]))
print("B", B)

