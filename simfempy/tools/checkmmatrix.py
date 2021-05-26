import scipy.sparse as sparse

def diffudsionForMMatrix(A):
    A = A.tocsr()
    Am = A.maximum(0)
    Am -= sparse.dia_matrix((Am.diagonal(),0),A.shape)
    Am = Am.maximum(Am.T)
    Am = Am + Am.T
    diagp = Am.sum(axis=1).ravel()
    D = sparse.dia_matrix((diagp,0),A.shape)
    # print(f"{D.diagonal()=}\n {A.minimum(0).todense()=}")
    return D - Am


def checkMmatrix(Ain, tol =1e-14):
    # test !!
    # A = (AI + AB + AR).tocoo()
    # for i, j, v in zip(A.row, A.col, A.data):
    #     print "row = %d, column = %d, value = %s" % (i, j, v)
    A = Ain.tolil()
    i = 0
    wrongsign, wrongdiagdom = [], []
    for rowi, dat in zip(A.rows, A.data):
        sum = 0.0
        for j, aij in zip(rowi, dat):
            if i == j:
                diag = aij
            else:
                if aij > tol:
                    wrongsign.append((i,j,aij))
                    # raise ValueError("Not a M-matrix i=%d, j=%d, aij=%g" % (i, j, aij))
                sum -= aij
        if 'diag' in locals() and diag < sum -tol:
            wrongdiagdom.append((i, *zip(rowi, dat)))
            # raise ValueError("Not a M-matrix diag-sum=%g" % (diag-sum))
        i += 1
    return wrongsign, wrongdiagdom
