import numpy as np

class MoveData():
    def __init__(self, n, d, cells=True, deltas=True):
        self.mus = np.empty(shape=(n, d + 1))
        if deltas: self.deltas = np.empty(n)
        if cells:
            self.imax = np.iinfo(np.uint).max
            self.cells = np.full(n, self.imax, dtype=np.uint)
    def mask(self):
        return self.cells != self.imax
    def plot(self, mesh, beta, type='nodes'):
        assert mesh.dimension==2
        import matplotlib.pyplot as plt
        from simfempy.meshes import plotmesh
        celldata = {f"beta": [beta[:, i] for i in range(mesh.dimension)]}
        plotmesh.meshWithData(mesh, quiver_data=celldata, plotmesh=True)
        ax = plt.gca()
        if type=='nodes':
            mask = self.mask()
            ax.plot(mesh.points[mask, 0], mesh.points[mask, 1], 'xr')
            mp = np.einsum('nik,ni->nk', mesh.points[mesh.simplices[self.cells[mask]]], self.mus[mask])
            ax.plot(mp[:, 0], mp[:, 1], 'xb')
        elif type=='midpoints':
            ax.plot(mesh.pointsc[:, 0], mesh.pointsc[:, 1], 'xr')
            mp = np.einsum('nik,ni->nk', mesh.points[mesh.simplices], self.mus)
            ax.plot(mp[:, 0], mp[:, 1], 'xb')
        plt.show()


#=================================================================#
def _coef_beta_in_simplex(i, mesh, beta):
    d = mesh.dimension
    s = mesh.simplices[i]
    p = mesh.points[s][:,:d]
    a = p[1:]-p[0]
    betacoef = np.linalg.solve(a.T,beta)
    # print(f"{p=} {a=} {betacoef=}")
    return betacoef

#=================================================================#
def _move_in_simplex(i, lamb, betacoef):
    coef = np.full_like(betacoef, np.inf)
    mn = betacoef<0
    mp = betacoef>0
    lam = lamb[1:]
    coef[mn] = -lam[mn]/betacoef[mn]
    coef[mp] = (1-lam[mp])/betacoef[mp]
    delta = np.min(coef)
    bs = np.sum(betacoef)
    if bs>0: delta = np.min((delta,lamb[0]/bs))
    mu = np.empty_like(lamb)
    mu[1:] = lam + delta*betacoef
    mu[0] = 1-np.sum(mu[1:])
    # if delta>0: print(f"{betacoef=}  {lam=} {coef=} {delta=} {mu=}")
    return delta, mu

#=================================================================#
def move_points(mesh, beta):
    d, nn, nc = mesh.dimension, mesh.nnodes, mesh.ncells
    assert beta.shape == (nc, d)
    lambdas = np.eye(d+1)
    md = MoveData(nn, d)
    for i in range(mesh.ncells):
        betacoef = _coef_beta_in_simplex(i, mesh, beta[i])
        for ipl in range(d+1):
            delta, mu = _move_in_simplex(i, lambdas[ipl], betacoef)
            if delta>0:
                ip = mesh.simplices[i, ipl]
                md.cells[ip] = i
                md.deltas[ip] = delta
                md.mus[ip] = mu
    return md
#=================================================================#
def move_midpoints(mesh, beta, extreme=False):
    d, nn, nc = mesh.dimension, mesh.nnodes, mesh.ncells
    assert beta.shape == (nc, d)
    lambdas = np.ones(d+1)/(d+1)
    md = MoveData(nc, d, cells=False, deltas=not extreme)
    for i in range(mesh.ncells):
        betacoef = _coef_beta_in_simplex(i, mesh, beta[i])
        delta, mu = _move_in_simplex(i, lambdas, betacoef)
        assert delta>0
        if not extreme: md.deltas[i] = delta
        md.mus[i] = mu
    if extreme:
        ind = np.argmax(md.mus,axis=1)
        md.mus.fill(0)
        np.put_along_axis(md.mus, np.expand_dims(ind,axis=1), 1, axis=1)
    return md
