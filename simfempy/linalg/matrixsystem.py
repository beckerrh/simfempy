import numpy as np
import scipy.sparse as sparse
from simfempy import tools, linalg

#=================================================================#
class MatrixSystem():
    """
    """
    def __repr__(self):
        for i in range(self.nblocks):
            for j in range(self.nblocks):
                print(f"{i,j=} {self.matrix_of_couple[i,j]} {self.transpose_of_couple[i,j]} {self.scale_of_couple[i,j]}")
        raise ValueError("not written")
    def __init__(self, vectorview, matrices, positions, types=None):
        self.vectorview = vectorview
        self.matrices = matrices
        self.positions = positions
        self.stack_storage = vectorview.stack_storage
        self.types = types
        self.nblocks = len(vectorview.ncomps)
        self.matrix_of_couple = -1*np.ones(shape=(self.nblocks,self.nblocks), dtype=int)
        self.transpose_of_couple = np.zeros(shape=(self.nblocks,self.nblocks), dtype=bool)
        self.scale_of_couple = np.ones(shape=(self.nblocks,self.nblocks), dtype=float)
        assert len(matrices) == len(positions)
        for ip, mat in enumerate(matrices):
            positions_of_mat = positions[ip]
            for p_of_m in positions_of_mat:
                assert isinstance(p_of_m,dict)
                i,j = p_of_m.pop('pos')
                self.matrix_of_couple[i,j] = ip
                self.transpose_of_couple[i,j] = p_of_m.pop('trp', False)
                self.scale_of_couple[i,j] = p_of_m.pop('scl', 1.0)
                if len(p_of_m.keys())!=0:
                    raise ValueError(f"unused keys {p_of_m.keys()}")
    def to_single_matrix(self):
        all = np.empty(shape=(self.nblocks, self.nblocks), dtype=object)
        for i in range(self.nblocks):
            for j in range(self.nblocks):
                index_ij = self.matrix_of_couple[i,j]
                if index_ij==-1:
                    all[i,j] = None
                    continue
                if self.transpose_of_couple[i,j]:
                    all[i,j] = self.scale_of_couple[i,j]*self.matrices[index_ij].T
                else:
                    all[i,j] = self.scale_of_couple[i,j]*self.matrices[index_ij]
        return sparse.block_array(all, format='csr')

