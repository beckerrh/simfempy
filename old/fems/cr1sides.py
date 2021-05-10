def computeMatrixTransportUpwind(self, bdrylumped):
    dim, dV, nfaces = self.mesh.dimension, self.mesh.dV, self.mesh.nfaces
        innerfaces, foc = self.mesh.innerfaces, self.mesh.facesOfCells
        mus, deltas = self.md.mus, self.md.deltas
        print(f"{innerfaces.shape=} {dim=} {self.mesh.nfaces=}")
        ci0 = self.mesh.cellsOfInteriorFaces[:, 0]
        ci1 = self.mesh.cellsOfInteriorFaces[:, 1]
        normalsS = self.mesh.normals[innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        betartin = self.betart[innerfaces]
        # A = -self.computeMatrixTransport(type='centered').T
        cols0 = foc[ci0].ravel()
        cols1 = foc[ci1].ravel()
        infaces = np.arange(nfaces)[innerfaces]
        rows = np.repeat(infaces, dim + 1).ravel()
    # centered
    # matloc = np.full(shape=(dim+1),fill_value=1/(dim+1))
    # mat = np.einsum('n,k->nk', self.betart[innerfaces]*dS, matloc).ravel()
    # A = sparse.coo_matrix((mat, (rows, cols1)), shape=(nfaces, nfaces))
    # A -= sparse.coo_matrix((mat, (rows, cols0)), shape=(nfaces, nfaces))

    A = sparse.dok_matrix((nfaces, nfaces))

    choice = 2
    if choice == 1:
        mat0 = np.einsum('n,nk->nk', betartin * dS, 1 - dim * mus[ci0]).ravel()
        mat1 = np.einsum('n,nk->nk', betartin * dS, 1 - dim * mus[ci1]).ravel()
        A += sparse.coo_matrix((mat1, (rows, cols1)), shape=(nfaces, nfaces))
        A -= sparse.coo_matrix((mat0, (rows, cols0)), shape=(nfaces, nfaces))
        # bdry
        faces = self.mesh.bdryFaces()
        normalsS = self.mesh.normals[faces]
        dS = linalg.norm(normalsS, axis=1)
        cof = self.mesh.cellsOfFaces[faces, 0]
        cols = foc[cof].ravel()
        rows = np.repeat(faces, dim + 1).ravel()
        matloc = np.full(shape=(dim + 1), fill_value=1 / (dim + 1))
        # mat = np.einsum('n,k->nk', self.betart[faces] * dS, matloc).ravel()
        mat = np.einsum('n,nk->nk', self.betart[faces] * dS, 1 - dim * mus[cof]).ravel()
        A -= sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces))
        A += self.computeBdryMassMatrix(coeff=np.maximum(self.betart, 0))
        A += self.computeMatrixJump(self.betart, mode='dual')
        return A.tocsr()
    else:
        A += sparse.coo_matrix((np.maximum(betartin, 0) * dS, (infaces, infaces)), shape=(nfaces, nfaces))
        A -= sparse.coo_matrix((np.minimum(betartin, 0) * dS, (infaces, infaces)), shape=(nfaces, nfaces))
        mat0 = np.einsum('n,nk->nk', np.maximum(betartin, 0) * dS, 1 - dim * mus[ci0]).ravel()
        mat1 = np.einsum('n,nk->nk', np.minimum(betartin, 0) * dS, 1 - dim * mus[ci1]).ravel()
        A -= sparse.coo_matrix((mat0, (rows, cols0)), shape=(nfaces, nfaces))
        A += sparse.coo_matrix((mat1, (rows, cols1)), shape=(nfaces, nfaces))

        w1, w2 = simfempy.tools.checkmmatrix.checkMmatrix(A)
        print(f"A {w1=}\n{w2=}")

        # bdry
        faces = self.mesh.bdryFaces()
        normalsS = self.mesh.normals[faces]
        dS = linalg.norm(normalsS, axis=1)
        cof = self.mesh.cellsOfFaces[faces, 0]
        cols = foc[cof].ravel()
        rows = np.repeat(faces, dim + 1).ravel()
        # matloc = np.full(shape=(dim + 1), fill_value=1 / (dim + 1))
        # mat = np.einsum('n,k->nk', self.betart[faces] * dS, matloc).ravel()

        # mat = np.einsum('n,nk->nk', -np.minimum(self.betart[faces],0) * dS, 1-dim*mus[cof]).ravel()
        # B = sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces))
        # B = sparse.coo_matrix((np.absolute(self.betart[faces]) * dS, (faces, faces)), shape=(nfaces, nfaces))
        # print(f"{B.todense()=}")
        B = self.computeBdryMassMatrix(coeff=np.maximum(self.betart, 0))
        B -= sparse.coo_matrix((self.betart[faces] * dS, (faces, faces)), shape=(nfaces, nfaces))
        # matloc = np.full(shape=(dim + 1), fill_value=1 / (dim + 1))
        # mat = np.einsum('n,k->nk', np.maximum(self.betart[faces],0) * dS, matloc).ravel()
        # # mat = np.einsum('n,nk->nk', np.maximum(self.betart[faces],0) * dS, 1-dim*mus[cof]).ravel()
        # B -=  sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces))
        # print(f"{B.todense()=}")
        w1, w2 = simfempy.tools.checkmmatrix.checkMmatrix(B)
        print(f"B {w1=} {w2=}")
        # print(f"{faces=}\n{-np.minimum(self.betart[faces],0)=}\n{(1-dim*mus[cof])=}")
        # mat = np.einsum('n,nk->nk', -np.minimum(self.betart[faces],0) * dS, 1-dim*mus[cof]).ravel()
        # C = sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces))
        # C = sparse.coo_matrix((-np.minimum(self.betart[faces],0) * dS, (faces, faces)), shape=(nfaces, nfaces))
        # w1, w2 = simfempy.tools.checkmmatrix.checkMmatrix(C)
        # print(f"C {w1=} {w2=}")
        # # import matplotlib.pyplot as plt
        # # ax = simfempy.meshes.plotmesh.plotmeshWithNumbering(self.mesh, sides=True)
        # # self.md.plot(self.mesh, self.beta, type='midpoints', ax=ax)
        # # plt.show()
        # # raise NotImplemented()
        # A += B + C
        A += B
        A += self.computeMatrixJump(self.betart, mode='dual')
        w1, w2 = simfempy.tools.checkmmatrix.checkMmatrix(A)
        print(f"A {w1=}\n{w2=}")
        print(f"{A.diagonal()=}")
        return A.tocsr()

    # centered=True
    # if centered:
    #     rows0 = foc[ci0].ravel()
    #     rows1 = foc[ci1].ravel()
    #     cols = np.repeat(np.arange(nfaces)[innerfaces],dim+1).ravel()
    #     matloc = np.full(shape=(dim+1),fill_value=1/(dim+1))
    #     mat = np.einsum('n,k->nk', self.betart[innerfaces]*dS, matloc).ravel()
    #     # print(f"{rows0.shape=} {cols.shape=} {mat.shape=}")
    #     # A += sparse.coo_matrix((mat, (rows0, cols)), shape=(nfaces, nfaces))
    #     # A -= sparse.coo_matrix((mat, (rows1, cols)), shape=(nfaces, nfaces))
    #     A -= sparse.coo_matrix((mat, (cols, rows0)), shape=(nfaces, nfaces))
    #     A += sparse.coo_matrix((mat, (cols, rows1)), shape=(nfaces, nfaces))
    #     faces = self.mesh.bdryFaces()
    #     normalsS = self.mesh.normals[faces]
    #     dS = linalg.norm(normalsS, axis=1)
    #     rows = foc[self.mesh.cellsOfFaces[faces, 0]].ravel()
    #     cols = np.repeat(faces, dim + 1).ravel()
    #     matloc = np.full(shape=(dim + 1), fill_value=1 / (dim + 1))
    #     mat = np.einsum('n,k->nk', self.betart[faces] * dS, matloc).ravel()
    #     A -= sparse.coo_matrix((mat, (cols, rows)), shape=(nfaces, nfaces))
    # else:
    #     rows0 = foc[ci0].ravel()
    #     rows1 = foc[ci1].ravel()
    #     cols = np.repeat(np.arange(nfaces)[innerfaces],dim+1).ravel()
    #     matloc = np.full(shape=(dim+1),fill_value=1/(dim+1))
    #     mat = np.einsum('n,nk->nk', self.betart[innerfaces]*dS, 1-dim*self.md.mus[ci0]).ravel()
    #     # print(f"{rows0.shape=} {cols.shape=} {mat.shape=}")
    #     A += sparse.coo_matrix((mat, (rows0, cols)), shape=(nfaces, nfaces))
    #     mat = np.einsum('n,nk->nk', self.betart[innerfaces]*dS, 1-dim*self.md.mus[ci1]).ravel()
    #     A -= sparse.coo_matrix((mat, (rows1, cols)), shape=(nfaces, nfaces))
    #     faces = self.mesh.bdryFaces()
    #     normalsS = self.mesh.normals[faces]
    #     dS = linalg.norm(normalsS, axis=1)
    #     rows = foc[self.mesh.cellsOfFaces[faces, 0]].ravel()
    #     cols = np.repeat(faces, dim + 1).ravel()
    #     ci = self.mesh.cellsOfFaces[faces,0]
    #     mat = np.einsum('n,nk->nk', self.betart[faces] * dS, 1-dim*self.md.mus[ci]).ravel()
    #     A += sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces))
    #     B = self.computeMatrixTransport(type='supg')
    #     if not np.allclose(B.A, A.tocsr().A):
    #         raise ValueError(f"{B.todense()=}\n{A.todense()=}")
    A += self.computeMatrixJump(self.betart, adj=True)
    A += self.computeBdryMassMatrix(coeff=np.maximum(self.betart, 0), lumped=bdrylumped)
    return A.tocsr()

    # rows0 = foc[ci0].ravel()
    # rows1 = foc[ci1].ravel()
    # faceind = np.arange(nfaces)[innerfaces]
    # cols = np.repeat(faceind, dim+1).ravel()
    # matloc = np.full(shape=(dim+1),fill_value=1/(dim+1))
    #
    # betadS = np.maximum(self.betart[self.mesh.innerfaces], 0)*dS
    # mat = np.einsum('n,k->nk', betadS, matloc).ravel()
    # A += sparse.coo_matrix((betadS, (faceind, faceind)), shape=(nfaces, nfaces))
    # A -= sparse.coo_matrix((mat, (rows1, cols)), shape=(nfaces, nfaces))
    #
    # betadS = np.minimum(self.betart[self.mesh.innerfaces], 0)*dS
    # mat = np.einsum('n,k->nk', betadS, matloc).ravel()
    # A += sparse.coo_matrix((mat, (rows0, cols)), shape=(nfaces, nfaces))
    # A -= sparse.coo_matrix((betadS, (faceind, faceind)), shape=(nfaces, nfaces))
    # rows0 = np.repeat(foc[ci0],dim+1).ravel()
    # cols0 = np.tile(foc[ci0],dim+1).ravel()
    # rows1 = np.repeat(foc[ci1],dim+1).ravel()
    # cols1 = np.tile(foc[ci1],dim+1).ravel()
    # matloc = np.full(shape=(dim+1,dim+1),fill_value=1/(dim+1))
    # mat = np.einsum('n,kl->nkl', np.maximum(self.betart[self.mesh.innerfaces], 0)*dS, matloc).ravel()
    # A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(nfaces, nfaces))
    # A -= sparse.coo_matrix((mat, (rows1, cols0)), shape=(nfaces, nfaces))
    # mat = np.einsum('n,kl->nkl', np.minimum(self.betart[self.mesh.innerfaces], 0)*dS, matloc).ravel()
    # # print(f"{A.diagonal()=}")
    # A += sparse.coo_matrix((mat, (rows0, cols1)), shape=(nfaces, nfaces))
    # A -= sparse.coo_matrix((mat, (rows1, cols1)), shape=(nfaces, nfaces))
    # bdry
    # A += self.computeMatrixJump(self.betart)

    # print(f"{A.diagonal()=}")
