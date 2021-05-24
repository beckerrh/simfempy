import numpy as np

class MoveData():
    def __init__(self, n, d, cells=True, deltas=True, second=False):
        self.mus = np.empty(shape=(n, d + 1))
        if deltas: self.deltas = np.empty(n)
        if cells:
            self.imax = np.iinfo(np.uint).max
            self.cells = np.full(n, self.imax, dtype=np.uint)
        if second:
            assert deltas and cells
            self.deltas2 = np.empty(n)
            self.cells2 = np.full(n, self.imax, dtype=np.uint)
            self.mus2 = np.empty(shape=(n, d + 1))
    def mask(self):
        return self.cells != self.imax
    def mask2(self):
        return self.cells2 != self.imax
    def maskonly1(self):
        return (self.cells != self.imax) & (self.cells2 == self.imax)
    def plot(self, mesh, beta, type='nodes', ax=None):
        import matplotlib.pyplot as plt
        if ax is None: ax = plt.gca()
        assert mesh.dimension==2
        from simfempy.meshes import plotmesh
        celldata = {f"beta": [beta[:, i] for i in range(mesh.dimension)]}
        plotmesh.meshWithData(mesh, quiver_data=celldata, plotmesh=True, ax=ax)
        if type in['nodes','sides']:
            m = self.mask()
            if type=='node':
                ax.plot(mesh.points[m, 0], mesh.points[m, 1], 'or')
            else:
                ax.plot(mesh.pointsf[m, 0], mesh.pointsf[m, 1], 'or')
            mp = np.einsum('nik,ni->nk', mesh.points[mesh.simplices[self.cells[m]]], self.mus[m])
            ax.plot(mp[:, 0], mp[:, 1], '+b')
            # mp = np.einsum('nik,ni->nk', mesh.pointsf[mesh.facesOfCells[self.cells[m]]], 1-2*self.mus[m])
            # ax.plot(mp[:, 0], mp[:, 1], 'Dm')
            if hasattr(self, 'cells2'):
                m2 = self.mask2()
                mp = np.einsum('nik,ni->nk', mesh.points[mesh.simplices[self.cells2[m2]]], self.mus2[m2])
                ax.plot(mp[:, 0], mp[:, 1], 'xy')
        elif type=='midpoints':
            ax.plot(mesh.pointsc[:, 0], mesh.pointsc[:, 1], '+r')
            mp = np.einsum('nik,ni->nk', mesh.points[mesh.simplices], self.mus)
            ax.plot(mp[:, 0], mp[:, 1], 'xb')


#=================================================================#
def _coef_mu_in_neighbor(mesh, ic2, ic, mu):
    d = mesh.dimension
    s = mesh.simplices[ic]
    p = mesh.points[s][:,:d]
    s2 = mesh.simplices[ic2]
    p2 = mesh.points[s2][:,:d]
    a2 = p2[1:]-p2[0]
    mu2 = np.empty_like(mu)
    b = p.T@mu - p2[0]
    mu2[1:] = np.linalg.solve(a2.T,b)
    mu2[0] = 1 - np.sum(mu2[1:])
    assert np.allclose(p.T@mu, p2.T@mu2, rtol=1e-12, atol=1e-14)
    return mu2
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
def _move_in_simplex_opt(lamb, betacoef, bound=1.0):
    # maximise delta under consraint 0\le \mu\le bound
    assert np.all(lamb<=bound)
    coef = np.full_like(betacoef, 0)
    mn = betacoef<0
    mp = betacoef>0
    lam = lamb[1:]
    coef[mn] = -lam[mn]/betacoef[mn]
    coef[mp] = (bound-lam[mp])/betacoef[mp]
    delta = np.min(coef)
    bs = np.sum(betacoef)
    if bs>0: delta = np.min((delta,lamb[0]/bs))
    if bs<0: delta = np.min((delta,(lamb[0]-bound)/bs))
    mu = np.empty_like(lamb)
    # print(f"{delta=}")
    mu[1:] = lam + delta*betacoef
    mu[0] = 1-np.sum(mu[1:])
    # if delta>0: print(f"{betacoef=}  {lam=} {coef=} {delta=} {mu=}")
    return delta, mu
#=================================================================#
def _move_in_simplex_candidates(i, mesh, beta, lamb, type):
    # maximise scalar product of candidates with beta
    dim, s = mesh.dimension, mesh.simplices[i]
    points = mesh.points[s,:dim]
    ncand = dim+1
    if type=='all': ncand *= 2
    candidates = np.full((ncand,dim+1), 1/dim)
    for d in range(dim + 1): candidates[d,d] = 0
    if type=='all':
        candidates[dim + 1:, :] = 1/dim**2
        for d in range(dim + 1): candidates[dim + 1+d, d] = (dim-1)/dim
    # print(f"{candidates=}")

    deltas = np.empty(ncand)
    for d in range(ncand): deltas[d] = np.linalg.norm(points.T@(candidates[d]-lamb))
    coef = np.empty(ncand)
    # print(f"{().shape=} {points.shape=} {beta.shape=}")
    for d in range(ncand): coef[d] = np.einsum('j,jk,k->',candidates[d,:]-lamb, points, beta)
    istar = np.argmax(coef/deltas)
    # if istar>dim: print(f"{coef=} {istar=}")
    return deltas[istar], candidates[istar]
#=================================================================#
def move_sides(mesh, beta, second=False):
    d, ns, nc = mesh.dimension, mesh.nfaces, mesh.ncells
    assert beta.shape == (nc, d)
    md = MoveData(ns, d, second=second)
    for i in range(mesh.ncells):
        betacoef = _coef_beta_in_simplex(i, mesh, beta[i])
        for ipl in range(d+1):
            lam = np.ones(d+1)/d
            lam[ipl] = 0
            delta, mu = _move_in_simplex_opt(lam, betacoef)
            print(f"{delta=} {mu=}")
            if delta>0:
                ip = mesh.facesOfCells[i, ipl]
                md.cells[ip] = i
                md.deltas[ip] = delta
                md.mus[ip] = mu
    return md
#=================================================================#
def move_nodes(mesh, beta, second=False):
    d, nn, nc = mesh.dimension, mesh.nnodes, mesh.ncells
    assert beta.shape == (nc, d)
    lambdas = np.eye(d+1)
    md = MoveData(nn, d, second=second)
    for i in range(mesh.ncells):
        betacoef = _coef_beta_in_simplex(i, mesh, beta[i])
        for ipl in range(d+1):
            delta, mu = _move_in_simplex_opt(lambdas[ipl], betacoef)
            if delta>0:
                ip = mesh.simplices[i, ipl]
                md.cells[ip] = i
                md.deltas[ip] = delta
                md.mus[ip] = mu
    ind = np.argmin(md.mus, axis=1)
    # print(f"{md.mus.shape=} {ind.shape=}")
    np.put_along_axis(md.mus,ind[:,np.newaxis],0,axis=1)
    if second:
        for ip in range(nn):
            ic = md.cells[ip]
            if ic == md.imax: continue
            indf = mesh.facesOfCells[ic,md.mus[ip]==0]
            ic2 = mesh.cellsOfFaces[indf,0][0]
            if ic2==ic: ic2 = mesh.cellsOfFaces[indf,1][0]
            if ic2<0: continue
            # print(f"{ic2=}")
            mu2 =_coef_mu_in_neighbor(mesh, ic2, ic, md.mus[ip])
            betacoef = _coef_beta_in_simplex(ic2, mesh, beta[ic])
            delta, mu = _move_in_simplex_opt(mu2, betacoef)
            # print(f"{delta=} {betacoef=} {md.mus[ip]} {mu=}")
            if delta > 0.1*md.deltas[ip]/np.sqrt(2):
                md.cells2[ip] = ic2
                md.deltas2[ip] = delta
                md.mus2[ip] = mu
    return md
#=================================================================#
def move_midpoints(mesh, beta, extreme=False, candidates=None, bound=1):
    d, nn, nc = mesh.dimension, mesh.nnodes, mesh.ncells
    assert beta.shape == (nc, d)
    lambdas = np.ones(d+1)/(d+1)
    md = MoveData(nc, d, cells=False, deltas=not extreme)
    if candidates:
        for i in range(mesh.ncells):
            md.deltas[i], md.mus[i] = _move_in_simplex_candidates(i, mesh, beta[i], lambdas, candidates)
    else:
        for i in range(mesh.ncells):
            betacoef = _coef_beta_in_simplex(i, mesh, beta[i])
            delta, mu = _move_in_simplex_opt(lambdas, betacoef, bound=bound)
            # print(f"{delta=} {mu=}")
            # assert delta>0
            if not extreme: md.deltas[i] = delta
            md.mus[i] = mu
        if extreme:
            ind = np.argmax(md.mus,axis=1)
            m = (np.take_along_axis(md.mus, np.expand_dims(ind,axis=1), axis=1)>=0.7).ravel()
            md.mus[m,:] = 0
            md.mus[m,ind[m]] = 1
    return md
#=================================================================#
def move_midpoint_to_neighbour(mesh, betart):
    import matplotlib.pyplot as plt
    from simfempy.meshes import plotmesh
    d, nn, ns, nc = mesh.dimension, mesh.nnodes, mesh.nfaces, mesh.ncells
    assert betart.shape[0] == ns
    md = MoveData(nc, d, cells=True)
    cof, foc = mesh.cellsOfFaces, mesh.facesOfCells
    ind = np.argmax(betart[foc]*mesh.sigma,axis=1)
    print(f"{ind=} {nn=} {ns=} {nc=}")
    print(f"{ind.shape=} {cof.shape=} {foc.shape=}")
    fi = np.take_along_axis(foc, ind[:,np.newaxis], axis=1).ravel()
    ci = cof[fi]
    # print(f"{fi=}")
    # print(f"{cof[fi]=}")
    # mb = cof[fi,1]==-1
    # print(f"{mb.shape=}\n{mb=}")
    # print(f"{ci[mb,0]=}")
    npa = np.arange(nc)
    m2 = ci!=npa[:,np.newaxis]
    md.cells = ci[m2]
    m = md.cells == -1
    md.cells[m] = npa[m]
    # print(f"{md.cells=}")
    mp = np.full(d+1,fill_value=1/(d+1))
    for i in range(mesh.ncells):
        md.mus[i] = _coef_mu_in_neighbor(mesh, md.cells[i], i, mp)
    # print(f"{md.mus=}")
    # plotmesh.plotmeshWithNumbering(mesh, sides=True)
    # plt.plot(mesh.pointsc[:, 0], mesh.pointsc[:, 1], 'or')
    # mp = np.einsum('nik,ni->nk', mesh.points[mesh.simplices[md.cells]], md.mus)
    # plt.plot(mp[:, 0], mp[:, 1], 'xb',alpha=0.9)
    # plt.show()
    return md
