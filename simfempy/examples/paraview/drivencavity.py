# state file generated using paraview version 5.9.1

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
renderView1.ViewSize = [2242, 1496]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.5, 0.5, 0.5]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-2.1802040995430074, -1.3689428935375778, 1.2209097503810526]
renderView1.CameraFocalPoint = [0.4999999999999995, 0.500000000000001, 0.5000000000000009]
renderView1.CameraViewUp = [0.10119623761043704, 0.22838002365876958, 0.9682984489748561]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.8660254037844386
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(2242, 1496)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
drivenCavity3dvtu = XMLUnstructuredGridReader(registrationName='drivenCavity3d.vtu', FileName=['/Users/becker/Programs/simfempy/drivenCavity3d.vtu'])
drivenCavity3dvtu.CellArrayStatus = ['P']
drivenCavity3dvtu.PointArrayStatus = ['V_0', 'V_1', 'V_2']
drivenCavity3dvtu.TimeArray = 'None'

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=drivenCavity3dvtu)
calculator1.ResultArrayName = 'V'
calculator1.Function = 'V_0*iHat+V_1*jHat+V_2*kHat'

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=calculator1)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.5, 0.21791778158879116, 0.5]
slice1.SliceType.Normal = [0.0, 1.0, 0.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.5, 0.5, 0.5]

# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=slice1,
    GlyphType='Arrow')
glyph1.OrientationArray = ['POINTS', 'V']
glyph1.ScaleArray = ['POINTS', 'V']
glyph1.ScaleFactor = 0.3
glyph1.GlyphTransform = 'Transform2'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from drivenCavity3dvtu
drivenCavity3dvtuDisplay = Show(drivenCavity3dvtu, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'V_2'
v_2LUT = GetColorTransferFunction('V_2')
v_2LUT.RGBPoints = [-0.3055438635809993, 0.231373, 0.298039, 0.752941, -0.034780728504573255, 0.865003, 0.865003, 0.865003, 0.23598240657185277, 0.705882, 0.0156863, 0.14902]
v_2LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'V_2'
v_2PWF = GetOpacityTransferFunction('V_2')
v_2PWF.Points = [-0.3055438635809993, 0.0, 0.5, 0.0, 0.23598240657185277, 1.0, 0.5, 0.0]
v_2PWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
drivenCavity3dvtuDisplay.Representation = 'Outline'
drivenCavity3dvtuDisplay.ColorArrayName = ['POINTS', 'V_2']
drivenCavity3dvtuDisplay.LookupTable = v_2LUT
drivenCavity3dvtuDisplay.SelectTCoordArray = 'None'
drivenCavity3dvtuDisplay.SelectNormalArray = 'None'
drivenCavity3dvtuDisplay.SelectTangentArray = 'None'
drivenCavity3dvtuDisplay.OSPRayScaleArray = 'V_0'
drivenCavity3dvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
drivenCavity3dvtuDisplay.SelectOrientationVectors = 'None'
drivenCavity3dvtuDisplay.ScaleFactor = 0.1
drivenCavity3dvtuDisplay.SelectScaleArray = 'None'
drivenCavity3dvtuDisplay.GlyphType = 'Arrow'
drivenCavity3dvtuDisplay.GlyphTableIndexArray = 'None'
drivenCavity3dvtuDisplay.GaussianRadius = 0.005
drivenCavity3dvtuDisplay.SetScaleArray = ['POINTS', 'V_0']
drivenCavity3dvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
drivenCavity3dvtuDisplay.OpacityArray = ['POINTS', 'V_0']
drivenCavity3dvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
drivenCavity3dvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
drivenCavity3dvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
drivenCavity3dvtuDisplay.ScalarOpacityFunction = v_2PWF
drivenCavity3dvtuDisplay.ScalarOpacityUnitDistance = 0.10413061241559457
drivenCavity3dvtuDisplay.OpacityArrayName = ['POINTS', 'V_0']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
drivenCavity3dvtuDisplay.ScaleTransferFunction.Points = [-0.15607901124981832, 0.0, 0.5, 0.0, 1.2248496966186877, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
drivenCavity3dvtuDisplay.OpacityTransferFunction.Points = [-0.15607901124981832, 0.0, 0.5, 0.0, 1.2248496966186877, 1.0, 0.5, 0.0]

# show data from slice1
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'P'
pLUT = GetColorTransferFunction('P')
pLUT.RGBPoints = [-0.25406850868022934, 0.231373, 0.298039, 0.752941, 0.011884899410226968, 0.865003, 0.865003, 0.865003, 0.2778383075006833, 0.705882, 0.0156863, 0.14902]
pLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = ['CELLS', 'P']
slice1Display.LookupTable = pLUT
slice1Display.SelectTCoordArray = 'None'
slice1Display.SelectNormalArray = 'None'
slice1Display.SelectTangentArray = 'None'
slice1Display.OSPRayScaleArray = 'Result'
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.SelectOrientationVectors = 'Result'
slice1Display.ScaleFactor = 0.1
slice1Display.SelectScaleArray = 'None'
slice1Display.GlyphType = 'Arrow'
slice1Display.GlyphTableIndexArray = 'None'
slice1Display.GaussianRadius = 0.005
slice1Display.SetScaleArray = ['POINTS', 'Result']
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityArray = ['POINTS', 'Result']
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [-0.12639338035797343, 0.0, 0.5, 0.0, 1.2172975418333962, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [-0.12639338035797343, 0.0, 0.5, 0.0, 1.2172975418333962, 1.0, 0.5, 0.0]

# show data from glyph1
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = [None, '']
glyph1Display.SelectTCoordArray = 'None'
glyph1Display.SelectNormalArray = 'None'
glyph1Display.SelectTangentArray = 'None'
glyph1Display.OSPRayScaleArray = 'Result'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'Result'
glyph1Display.ScaleFactor = 0.11089509502053262
glyph1Display.SelectScaleArray = 'None'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.GlyphTableIndexArray = 'None'
glyph1Display.GaussianRadius = 0.0055447547510266305
glyph1Display.SetScaleArray = ['POINTS', 'Result']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph1Display.OpacityArray = ['POINTS', 'Result']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph1Display.ScaleTransferFunction.Points = [-0.13732019728357697, 0.0, 0.5, 0.0, 1.0630968654724513, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph1Display.OpacityTransferFunction.Points = [-0.13732019728357697, 0.0, 0.5, 0.0, 1.0630968654724513, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pLUT in view renderView1
pLUTColorBar = GetScalarBar(pLUT, renderView1)
pLUTColorBar.Title = 'P'
pLUTColorBar.ComponentTitle = ''

# set color bar visibility
pLUTColorBar.Visibility = 1

# get color legend/bar for v_2LUT in view renderView1
v_2LUTColorBar = GetScalarBar(v_2LUT, renderView1)
v_2LUTColorBar.WindowLocation = 'UpperRightCorner'
v_2LUTColorBar.Title = 'V_2'
v_2LUTColorBar.ComponentTitle = ''

# set color bar visibility
v_2LUTColorBar.Visibility = 1

# show color legend
drivenCavity3dvtuDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'P'
pPWF = GetOpacityTransferFunction('P')
pPWF.Points = [-0.25406850868022934, 0.0, 0.5, 0.0, 0.2778383075006833, 1.0, 0.5, 0.0]
pPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# restore active source
SetActiveSource(slice1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')