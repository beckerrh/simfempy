# state file generated using paraview version 5.11.1
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1101, 974]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [1.0, 0.0, 0.5]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-5.510403254817618, 4.070379029890183, -3.9067823830586113]
renderView1.CameraFocalPoint = [1.0000000000000013, -1.9772886846001355e-15, 0.4999999999999989]
renderView1.CameraViewUp = [0.43993051783057924, -0.23478442972582148, -0.8667972144853945]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 2.29128784747792
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1101, 974)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
navierStokesvtu = XMLUnstructuredGridReader(registrationName='NavierStokes.vtu', FileName=['/Users/becker/Programs/simfempy/simfempy/examples/NavierStokes.vtu'])
navierStokesvtu.CellArrayStatus = ['P']
navierStokesvtu.PointArrayStatus = ['V_0', 'V_1', 'V_2']
navierStokesvtu.TimeArray = 'None'

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=navierStokesvtu)
calculator1.Function = 'V_0*iHat+V_1*jHat+V_2*kHat'

# create a new 'Slice With Plane'
sliceWithPlane1 = SliceWithPlane(registrationName='SliceWithPlane1', Input=calculator1)
sliceWithPlane1.PlaneType = 'Plane'
sliceWithPlane1.Level = -1

# init the 'Plane' selected for 'PlaneType'
sliceWithPlane1.PlaneType.Origin = [0.4277467349634633, -0.0007503514250675896, 0.5]
sliceWithPlane1.PlaneType.Normal = [0.0, 0.0, 1.0]

# create a new 'Glyph'
glyph2 = Glyph(registrationName='Glyph2', Input=sliceWithPlane1,
    GlyphType='Arrow')
glyph2.OrientationArray = ['POINTS', 'Result']
glyph2.ScaleArray = ['POINTS', 'No scale array']
glyph2.ScaleFactor = 0.4
glyph2.GlyphTransform = 'Transform2'

# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=calculator1,
    GlyphType='Arrow')
glyph1.OrientationArray = ['POINTS', 'Result']
glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1.ScaleFactor = 0.4
glyph1.GlyphTransform = 'Transform2'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from navierStokesvtu
navierStokesvtuDisplay = Show(navierStokesvtu, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'P'
pTF2D = GetTransferFunction2D('P')

# get color transfer function/color map for 'P'
pLUT = GetColorTransferFunction('P')
pLUT.TransferFunction2D = pTF2D
pLUT.RGBPoints = [0.00013160095827975711, 0.231373, 0.298039, 0.752941, 0.004551191489483337, 0.865003, 0.865003, 0.865003, 0.008970782020686918, 0.705882, 0.0156863, 0.14902]
pLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'P'
pPWF = GetOpacityTransferFunction('P')
pPWF.Points = [0.00013160095827975711, 0.0, 0.5, 0.0, 0.008970782020686918, 1.0, 0.5, 0.0]
pPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
navierStokesvtuDisplay.Representation = 'Wireframe'
navierStokesvtuDisplay.ColorArrayName = ['CELLS', 'P']
navierStokesvtuDisplay.LookupTable = pLUT
navierStokesvtuDisplay.SelectTCoordArray = 'None'
navierStokesvtuDisplay.SelectNormalArray = 'None'
navierStokesvtuDisplay.SelectTangentArray = 'None'
navierStokesvtuDisplay.OSPRayScaleArray = 'V_0'
navierStokesvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
navierStokesvtuDisplay.SelectOrientationVectors = 'None'
navierStokesvtuDisplay.ScaleFactor = 0.4
navierStokesvtuDisplay.SelectScaleArray = 'None'
navierStokesvtuDisplay.GlyphType = 'Arrow'
navierStokesvtuDisplay.GlyphTableIndexArray = 'None'
navierStokesvtuDisplay.GaussianRadius = 0.02
navierStokesvtuDisplay.SetScaleArray = ['POINTS', 'V_0']
navierStokesvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
navierStokesvtuDisplay.OpacityArray = ['POINTS', 'V_0']
navierStokesvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
navierStokesvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
navierStokesvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
navierStokesvtuDisplay.ScalarOpacityFunction = pPWF
navierStokesvtuDisplay.ScalarOpacityUnitDistance = 0.3951252478330957
navierStokesvtuDisplay.OpacityArrayName = ['POINTS', 'V_0']
navierStokesvtuDisplay.SelectInputVectors = [None, '']
navierStokesvtuDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
navierStokesvtuDisplay.ScaleTransferFunction.Points = [-0.009789847628199157, 0.0, 0.5, 0.0, 0.06785252092597915, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
navierStokesvtuDisplay.OpacityTransferFunction.Points = [-0.009789847628199157, 0.0, 0.5, 0.0, 0.06785252092597915, 1.0, 0.5, 0.0]

# show data from sliceWithPlane1
sliceWithPlane1Display = Show(sliceWithPlane1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sliceWithPlane1Display.Representation = 'Surface'
sliceWithPlane1Display.ColorArrayName = ['CELLS', 'P']
sliceWithPlane1Display.LookupTable = pLUT
sliceWithPlane1Display.SelectTCoordArray = 'None'
sliceWithPlane1Display.SelectNormalArray = 'None'
sliceWithPlane1Display.SelectTangentArray = 'None'
sliceWithPlane1Display.OSPRayScaleArray = 'Result'
sliceWithPlane1Display.OSPRayScaleFunction = 'PiecewiseFunction'
sliceWithPlane1Display.SelectOrientationVectors = 'Result'
sliceWithPlane1Display.ScaleFactor = 0.2
sliceWithPlane1Display.SelectScaleArray = 'None'
sliceWithPlane1Display.GlyphType = 'Arrow'
sliceWithPlane1Display.GlyphTableIndexArray = 'None'
sliceWithPlane1Display.GaussianRadius = 0.01
sliceWithPlane1Display.SetScaleArray = ['POINTS', 'Result']
sliceWithPlane1Display.ScaleTransferFunction = 'PiecewiseFunction'
sliceWithPlane1Display.OpacityArray = ['POINTS', 'Result']
sliceWithPlane1Display.OpacityTransferFunction = 'PiecewiseFunction'
sliceWithPlane1Display.DataAxesGrid = 'GridAxesRepresentation'
sliceWithPlane1Display.PolarAxes = 'PolarAxesRepresentation'
sliceWithPlane1Display.SelectInputVectors = ['POINTS', 'Result']
sliceWithPlane1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
sliceWithPlane1Display.ScaleTransferFunction.Points = [-0.0030818182946231324, 0.0, 0.5, 0.0, 0.038028544896319794, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
sliceWithPlane1Display.OpacityTransferFunction.Points = [-0.0030818182946231324, 0.0, 0.5, 0.0, 0.038028544896319794, 1.0, 0.5, 0.0]

# show data from glyph2
glyph2Display = Show(glyph2, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
glyph2Display.Representation = 'Surface'
glyph2Display.ColorArrayName = [None, '']
glyph2Display.SelectTCoordArray = 'None'
glyph2Display.SelectNormalArray = 'None'
glyph2Display.SelectTangentArray = 'None'
glyph2Display.OSPRayScaleArray = 'Result'
glyph2Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph2Display.SelectOrientationVectors = 'Result'
glyph2Display.ScaleFactor = 0.44037172794342044
glyph2Display.SelectScaleArray = 'None'
glyph2Display.GlyphType = 'Arrow'
glyph2Display.GlyphTableIndexArray = 'None'
glyph2Display.GaussianRadius = 0.02201858639717102
glyph2Display.SetScaleArray = ['POINTS', 'Result']
glyph2Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph2Display.OpacityArray = ['POINTS', 'Result']
glyph2Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph2Display.DataAxesGrid = 'GridAxesRepresentation'
glyph2Display.PolarAxes = 'PolarAxesRepresentation'
glyph2Display.SelectInputVectors = ['POINTS', 'Result']
glyph2Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph2Display.ScaleTransferFunction.Points = [-0.0004759272248873353, 0.0, 0.5, 0.0, 0.06570508697861333, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph2Display.OpacityTransferFunction.Points = [-0.0004759272248873353, 0.0, 0.5, 0.0, 0.06570508697861333, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pLUT in view renderView1
pLUTColorBar = GetScalarBar(pLUT, renderView1)
pLUTColorBar.Title = 'P'
pLUTColorBar.ComponentTitle = ''

# set color bar visibility
pLUTColorBar.Visibility = 1

# show color legend
navierStokesvtuDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
sliceWithPlane1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(glyph2)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')