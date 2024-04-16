import pathlib, sys
simfempypath = str(pathlib.Path(__file__).parent.parent.parent.parent)
sys.path.insert(0,simfempypath)
import simfempy.meshes.testmeshes as testmeshes
from simfempy.models.elliptic import Elliptic
import simfempy.models.problemdata
from simfempy.tools.comparemethods import CompareMethods


#----------------------------------------------------------------#
class EllipticApplicationWithExactSolution(simfempy.applications.application.Application):
    def __init__(self, dim, exactsolution, **kwargs):
        super().__init__(exactsolution=exactsolution)
        # self.exactsolution = exactsolution
        data = self.problemdata
        data.ncomp = 1
        colorsneu, colorsrob = [], []
        if dim == 1:
            self.defineGeometry = testmeshes.unitline
            colors = [10000,10001]
        elif dim == 2:
            self.defineGeometry = testmeshes.unitsquare
            colors = [1000, 1001, 1002, 1003]
            # colorsrob = [1002]
            # colorsneu = [1001]
        else:
            self.defineGeometry = testmeshes.unitcube
            colors = [100, 101, 102, 103, 104, 105]
            # colorsrob = [101]
            # colorsneu = [103]
        colorsdir = [col for col in colors if col not in colorsrob and col not in colorsneu]
        data.bdrycond.set("Dirichlet", colorsdir)
        data.bdrycond.set("Neumann", colorsneu)
        data.bdrycond.set("Robin", colorsrob)
        for col in colorsrob: data.bdrycond.param[col] = 100.
        data.params.scal_glob['kheat'] = kwargs.pop('kheat', 0.01)
        data.params.fct_glob['convection'] = ['0.8', '1.1']


#----------------------------------------------------------------#
def test(dim, exactsolution, paramsdict, modelargs, **kwargs):
    app = EllipticApplicationWithExactSolution(dim, exactsolution, **kwargs)
    comp =  CompareMethods(application=app, paramsdict=paramsdict, model=Elliptic, modelargs=modelargs, **kwargs)
    return comp.compare()

#================================================================#
if __name__ == '__main__':
    # TODO P1 with convection (centered) and nitsche wrong (!) termes de bord ?
    exactsolution = 'Linear'
    disc_params = {'dirichletmethod':'nitsche', 'convmethod':'lps', 'lpsparam':0.1}
    modelargs = {'fem':'cr1', 'disc_params':disc_params, 'mode':'linear', 'linearsolver':{'method':'pyamg', 'rtol':1e-12},'newton_rtol':1e-12}
    # modelargs = {'fem':'cr1', 'disc_params':disc_params, 'mode':'linear', 'linearsolver':'spsolve'}
    paramsdict = {'fem':['cr1','p1','rt0']}
    paramsdict = {'fem':['p1']}
    test(dim=2, exactsolution = exactsolution, paramsdict=paramsdict, modelargs=modelargs, plotsolution='True')
