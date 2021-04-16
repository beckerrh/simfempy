import numpy as np

#=================================================================#
def _coef_beta_in_simplex(i, mesh, beta):
    d = mesh.dimension
    s = mesh.simplices[i]
    p = mesh.points[s][:,:d]
    a = np.empty((d,d))
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
    assert beta.shape == (mesh.ncells, mesh.dimension)
    d, nn = mesh.dimension, mesh.nnodes
    lambdas = np.eye(d+1)
    imax = np.iinfo(np.uint).max
    print(f"{imax=}")
    cells, deltas, mus = np.full(nn, imax, dtype=np.uint), np.empty(nn), np.empty(shape=(nn,d+1))
    for i in range(mesh.ncells):
        betacoef = _coef_beta_in_simplex(i, mesh, beta[i])
        for ipl in range(d+1):
            delta, mu = _move_in_simplex(i, lambdas[ipl], betacoef)
            if delta>0:
                ip = mesh.simplices[i, ipl]
                cells[ip] = i
                deltas[ip] = delta
                mus[ip] = mu
    import matplotlib.pyplot as plt
    from simfempy.meshes import plotmesh
    celldata = {f"beta": [beta[:, i] for i in range(mesh.dimension)]}
    plotmesh.meshWithData(mesh, quiver_data=celldata, plotmesh=True)
    ax = plt.gca()
    mask = cells != imax
    # print(f"{cells=} {mask}")
    ax.plot(mesh.points[mask, 0], mesh.points[mask, 1], 'xr')

    mp = np.einsum('nik,ni->nk',mesh.points[mesh.simplices[cells[mask]]],mus[mask])
    print(f"{mp.shape=}")
    ax.plot(mp[:, 0], mp[:, 1], 'xb')

    # xd, ld, delta = self.downWind(beta)
    # ax.plot(xd[:, 0], xd[:, 1], 'xr')
    # xd, ld, delta = self.downWind(beta, method='supg2')
    # ax.plot(xd[:, 0], xd[:, 1], 'xb')
    plt.show()
