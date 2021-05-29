import numpy as np
import scipy.sparse as sparse

def addRankOne(A, x, y, relax=1):
    """
    Add rank-one matrix B such that 
    (A+B)x = y 
    B x = y - Ax
    """
    A = A.tocsr()
    indptr, indices = A.indptr, A.indices
    # print(f"{indptr=}\n {indices=}")
    y2 = y - A.dot(x)
    jsplit = np.split(indices,indptr)[1:-1]
    # print(f"{len(jsplit)=}")
    x2 = [y2[i]*x[j]/(0.001+x[j].dot(x[j])) for i,j in enumerate(jsplit)]
    B = np.concatenate(x2) 
    assert B.shape == A.data.shape
    A.data += relax*B
    # print(f"{np.concatenate(x2)=}")
    return A


# ------------------------------------------------------------------- #
if __name__ == '__main__':
    n = 6
    A = sparse.random(n,n, density=0.5,)
    x = np.random.random(6)    
    y = np.random.random(6)
    print(f"{A.todense()=}\n{x=}\n{y=}")
    A = addRankOne(A, x, y)
    print(f"{A.todense()=}\n{y-A.dot(x)=}")
