import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0,simfempypath)

import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.stokes import Stokes
import simfempy.applications.problemdata
from simfempy.tools.comparemethods import CompareMethods

#----------------------------------------------------------------#
def test(dim, **kwargs):
    exactsolution = kwargs.pop('exactsolution', 'Linear')
    data = simfempy.applications.problemdata.ProblemData()
    data.params.scal_glob['mu'] = kwargs.pop('mu', 1)
    data.params.scal_glob['navier'] = kwargs.pop('navier', 1)
    paramsdict = {}
    paramsdict['dirichletmethod'] = kwargs.pop('dirichletmethod', ['strong','nitsche'])
    if dim==2:
        data.ncomp=2
        createMesh = testmeshes.unitsquare
        colors = [1000,1001,1003]
        colorsneu = [1000]
        #TODO cl navier faux pour deux bords ?!
        colorsnav = [1001]
        colorsp = [1002]
    else:
        data.ncomp=3
        createMesh = testmeshes.unitcube
        colors = [100,101,102,103,104,105]
        colorsneu = [103]
        colorsnav = [105]
        colorsp = [101]
    colorsdir = [col for col in colors if col not in colorsnav and col not in colorsp and col not in colorsneu]
    if 'strong' in paramsdict['dirichletmethod']:
        colorsdir.append(*colorsnav)
        colorsdir.append(*colorsp)
        colorsnav=[]
        colorsp=[]
    data.bdrycond.set("Dirichlet", colorsdir)
    data.bdrycond.set("Neumann", colorsneu)
    data.bdrycond.set("Navier", colorsnav)
    data.bdrycond.set("Pressure", colorsp)
    data.postproc.set(name='bdrypmean', type='bdry_pmean', colors=colorsneu)
    data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=colorsdir)
    linearsolver = kwargs.pop('linearsolver', 'iter')
    applicationargs= {'problemdata': data, 'exactsolution': exactsolution, 'linearsolver': linearsolver}
    # applicationargs['mode'] = 'newton'
    comp =  CompareMethods(createMesh=createMesh, paramsdict=paramsdict, application=Stokes, applicationargs=applicationargs, **kwargs)
    return comp.compare()



#================================================================#
if __name__ == '__main__':
    # test(dim=2, exactsolution=[["x**2-y","-2*x*y+x**2"],"x*y"], dirichletmethod='nitsche', niter=6, plotsolution=False, linearsolver='iter_gcrotmk')
    # test(dim=3, exactsolution=[["x**2-y+2","-2*x*y+x**2","x+y"],"x*y*z"], dirichletmethod='nitsche', niter=5, plotsolution=False, linearsolver='iter_gcrotmk')
    # test(dim=2, exactsolution="Linear", niter=3, dirichletmethod='nitsche', plotsolution=True, linearsolver='umf')
    # test(dim=2, exactsolution="Quadratic", niter=7, dirichletmethod='nitsche', plotsolution=True, linearsolver='umf')
    # test(dim=2, exactsolution=[["1.0","0.0"],"10"], niter=3, dirichletmethod='nitsche', plotsolution=True, linearsolver='umf')
    # test(dim=3, exactsolution=[["-z","x","x+y"],"11"], niter=3, dirichletmethod=['nitsche'], plotsolution=False, linearsolver='umf')
    test(dim=2, exactsolution=[["-y","x"],"10"], niter=3, dirichletmethod='nitsche', plotsolution=False, linearsolver='umf')
