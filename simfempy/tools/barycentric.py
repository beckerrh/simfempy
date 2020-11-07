import numpy as np
import scipy, scipy.special
import itertools as it

# ------------------------------------- #
def tensor(d, k):
    A = np.ones(shape=k*[d+1])
    facd = np.prod(np.arange(d + 1, d + k + 1))
    # print(f"{np.arange(d + 1, d + k + 1)=} {facd=}")
    for i in it.product(np.arange(d+1), repeat=k):
        A[i] = np.prod(scipy.special.factorial(np.bincount(i)))/facd
    return A

# ------------------------------------- #
if __name__ == '__main__':
    print(f"{tensor(d=3, k=2)}")