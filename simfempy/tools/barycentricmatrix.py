import numpy as np
import scipy, scipy.special
import itertools as it

# ------------------------------------- #
def tensor(d, k):
    A = np.ones(shape=k*[d+1], dtype=int)
    for i in it.product(np.arange(d+1), repeat=k):
        # print(f"{i=} {np.bincount(i)=} {np.prod(scipy.special.factorial(np.bincount(i)))=}")
        A[i] = np.prod(scipy.special.factorial(np.bincount(i)))
    return A

# ------------------------------------- #
if __name__ == '__main__':
    # d = 2
    # k = 2
    # for i in sums(d + 1, k):
    #     # print(f"{i=} {scipy.special.factorial(np.array(i, dtype=int))}")
    #     print(f"{i=} {np.prod(scipy.special.factorial(i))}")
    # print(f"{matrix(2, 2)}")
    print(f"{tensor(d=3, k=3)}")