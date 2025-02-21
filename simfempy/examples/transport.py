import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from simfempy import fems, models, applications
from simfempy.tools.analyticalfunction import AnalyticalFunction

# ================================================================= #
class TransportModel(models.model.Model):
    def __init__(self, **kwargs):
        print(f"{kwargs=}")
        self.fem = kwargs.pop('fem','d0')
        self.linearsolver = kwargs.pop('linearsolver', 'pyamg')
        super().__init__(**kwargs)
    def createFem(self):
        self.fem = fems.d0.D0()
    def meshSet(self):
        self.mesh.constructInnerFaces()
        self.fem.setMesh(self.mesh)
        assert 'convection' in self.problemdata.params.fct_glob
        convection_given = self.problemdata.params.fct_glob['convection']
        if not isinstance(convection_given, list):
            p = "problemdata.params.fct_glob['convection']"
            raise ValueError(f"need '{p}' as a list of length dim of str or AnalyticalSolution")
        elif isinstance(convection_given[0], str):
            self.convection_fct = [AnalyticalFunction(expr=e) for e in convection_given]
        else:
            self.convection_fct = convection_given
            if not isinstance(convection_given[0], AnalyticalFunction):
                raise ValueError(f"convection should be given as 'str' and not '{type(convection_given[0])}'")
        if len(self.convection_fct) != self.mesh.dimension:
            raise ValueError(f"{self.mesh.dimension=} {self.problemdata.params.fct_glob['convection']=}")
        rt = fems.rt0.RT0(mesh=self.mesh)
        self.betart = rt.interpolate(self.convection_fct)

    def computeMatrix(self):
        import scipy.sparse as sparse
        nall = self.fem.nunknowns()
        A = sparse.coo_matrix((nall, nall))
        # mat = np.einsum('n,kl->nkl', dS*scale, massloc).reshape(-1)
        ci0 = self.mesh.cellsOfInteriorFaces[:,0]
        ci1 = self.mesh.cellsOfInteriorFaces[:,1]
        assert np.all(ci1>=0)
        normalsS = self.mesh.normals[self.mesh.innerfaces]
        dS = np.linalg.norm(normalsS, axis=1)
        faces = self.mesh.faces[self.mesh.innerfaces]
        matpos = np.maximum(self.betart[faces], 0)*dS
        matneg = np.minimum(self.betart[faces], 0)*dS
        A += sparse.coo_matrix((matpos, (ci0, ci0)), shape=(nall, nall))
        A += sparse.coo_matrix((-matpos, (ci1, ci0)), shape=(nall, nall))
        A += sparse.coo_matrix((matneg, (ci0, ci1)), shape=(nall, nall))
        A += sparse.coo_matrix((-matneg, (ci1, ci1)), shape=(nall, nall))

        colors = self.mesh.bdrylabels.keys()
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces, :dim]
        dS = np.linalg.norm(normalsS, axis=1)
        A += sparse.coo_matrix((np.maximum(self.betart, 0)*dS, (cells, cells)), shape=(nall, nall))

        # print(f"{self.fem.nunknowns()=} {A.data=}")
        return A
    def computeRhs(self, u=None):
        b = np.zeros(self.fem.nunknowns())
        if 'rhs' in self.problemdata.params.fct_glob:
            fp1 = self.fem.interpolate(self.problemdata.params.fct_glob['rhs'])
            # print(f"{fp1=}")
            # A = self.fem.computeMassMatrixSupg(self.xd)
            # b += A.dot(fp1)
            self.fem.massDot(b, fp1)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        if self.problemdata.solexact:
            # fex = self.fem.interpolate(self.problemdata.solexact)
            bdryfct = self.problemdata.bdrycond.fct
        else:
            bdryfct = {col:self.problemdata.solexact for col in colorsdir}
        self.fem.computeRhsBoundary(b, colorsdir, bdryfct, coeff=-np.minimum(self.betart, 0))
        print(f"{b=}")
        return b

# ================================================================= #
class TransportApplication(applications.application.Application):
    def __init__(self, dim, exactsolution, **kwargs):
        super().__init__(exactsolution=exactsolution)
    def defineGeometry(self, geom, h):
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=h)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
    def defineProblemData(self, problemdata):
        problemdata.bdrycond.ignore = True
        problemdata.params.fct_glob["convection"] = ["0.5", "1"]
        problemdata.bdrycond.set("Dirichlet", [1000, 1003])

#================================================================#
if __name__ == '__main__':
    dim, exactsolution = 2, None
    app = TransportApplication(dim, exactsolution)
    transport = TransportModel(application=app)
    result, u = transport.static(method = 'linear')
    # print(f"{result=}")
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"{transport.application.__class__.__name__} (static)", fontsize=16)
    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    transport.mesh.plot(fig=fig, outer=outer[0], bdry=True)
    # data = u.tovisudata()
    # data.update({'cell': {'k': transport.kheatcell}})
    # transport.mesh.plot(data=data, alpha=0.5, fig=fig, outer=outer[1])
    rt = fems.rt0.RT0(mesh=transport.mesh)
    beta = rt.toCell(transport.betart)
    print(f"{beta.shape=}")
    transport.mesh.plot(quiver_data={'beta':beta.T}, alpha=0.5, fig=fig, outer=outer[1])
    plt.show()