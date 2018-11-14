def checkMmatrix(Ain, tol =1e-14):
    # test !!
    # A = (AI + AB + AR).tocoo()
    # for i, j, v in zip(A.row, A.col, A.data):
    #     print "row = %d, column = %d, value = %s" % (i, j, v)
    A = Ain.tolil()
    i = 0
    for rowi, rowv in zip(A.rows, A.data):
        sum = 0.0
        for j, aij in zip(rowi, rowv):
            if i == j:
                diag = aij
            else:
                if aij > tol:
                    raise ValueError("Not a M-matrix i=%d, j=%d, aij=%g" % (i, j, aij))
                sum -= aij
        if diag < sum -tol:
            raise ValueError("Not a M-matrix diag-sum=%g" % (diag-sum))
        i += 1
