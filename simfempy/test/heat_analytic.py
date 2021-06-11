import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0,simfempypath)
import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.heat import Heat
import simfempy.applications.problemdata
from simfempy.test.test_analytic import test_analytic

#----------------------------------------------------------------#
def test(dim, **kwargs):
    data = simfempy.applications.problemdata.ProblemData()
    exactsolution = kwargs.pop('exactsolution', 'Linear')
    paramargs = {'fem': kwargs.pop('fem', ['p1','cr1'])}
    paramargs['dirichletmethod'] = kwargs.pop('dirichletmethod', ['nitsche','strong','new'])
    if 'convection' in kwargs:
        data.params.fct_glob['convection'] = kwargs.pop('convection')
        paramargs['convmethod'] = kwargs.pop('convmethod', ['supg'])
    data.params.scal_glob['kheat'] = kwargs.pop('kheat', 0.01)
    if dim==1:
        createMesh = testmeshes.unitline
        colors = [10000,10001]
        colorsrob = []
        colorsneu = [10001]
    elif dim==2:
        createMesh = testmeshes.unitsquare
        colors = [1000, 1001, 1002, 1003]
        colorsrob = []
        colorsneu = [1001]
    else:
        createMesh = testmeshes.unitcube
        colors = [100, 101, 102, 103, 104, 105]
        colorsrob = []
        colorsneu = [103]
    colorsdir = [col for col in colors if col not in colorsrob and col not in colorsneu]
    data.bdrycond.set("Dirichlet", colorsdir)
    data.bdrycond.set("Neumann", colorsneu)
    data.bdrycond.set("Robin", colorsrob)
    for col in colorsrob: data.bdrycond.param[col] = 11.
    data.postproc.set(name='bdrymean', type='bdry_mean', colors=colorsneu)
    data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=colorsdir[0])
    linearsolver = kwargs.pop('linearsolver', 'pyamg')
    applicationargs= {'problemdata': data, 'exactsolution': exactsolution, 'linearsolver': linearsolver, 'masslumpedbdry':False}
    # applicationargs['mode'] = 'newton'
    return test_analytic(application=Heat, createMesh=createMesh, paramargs=paramargs, applicationargs=applicationargs, **kwargs)

#================================================================#
if __name__ == '__main__':
    #TODO: pyamg in 1d/3d accel=bicgstab doesn't <ork
    #TODO: p1-new-robin wrong

    # test dirichletmethod
    test(dim=2, exactsolution = 'Linear', fem=['cr1'], niter=2, linearsolver='umf', dirichletmethod=['nitsche'], kheat=0.12, plotsolution=False)
    # test(dim=2, exactsolution = 'Linear', fem=['p1'], niter=3 , linearsolver='umf', dirichletmethod=['nitsche'], kheat=1, plotsolution=False)
    # test convection
    # test(dim=2, exactsolution = 'Linear', fem=['p1'], niter=6, h1=2, convection=["0.8","1.1"], convmethod=['upwalg','upw','upwsides'], dirichletmethod=['nitsche'], kheat=0.0, linearsolver='umf', plotsolution=True)
    # test(dim=2, exactsolution = 'Linear', fem=['cr1'], niter=4, h1=0.2, convection=["0.8","1.1"], convmethod=['upwalg','supg'], dirichletmethod=['nitsche'], kheat=0.0, linearsolver='pyamg', plotsolution=True)
    # test(dim=2, exactsolution = 'Linear', fem=['cr1'], niter=5, h1=2, convection=["1-x","1+y"], convmethod=['upw','supg'], dirichletmethod=['nitsche'], kheat=0.0, linearsolver='umf', plotsolution=True)
    # test(dim=2, exactsolution = 'Quadratic', fem=['p1'], niter=6, convection=["0.8","1.1"], convmethod=['upw'], dirichletmethod=['nitsche'], kheat=0.0001, linearsolver='pyamg')
