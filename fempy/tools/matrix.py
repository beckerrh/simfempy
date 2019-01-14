import numpy as np
import scipy.sparse as sparse

#=================================================================#
def is_symmetric(m):
    """Check if a sparse matrix is symmetric
        https://mail.python.org/pipermail/scipy-dev/2014-October/020117.html
        Parameters
        ----------
        m : sparse matrix

        Returns
        -------
        check : bool
    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, sparse.coo_matrix):
        m = sparse.coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False, "no_diag_sum", triu_no_diag.sum() - tril_no_diag.sum()

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check
