import os, sys
import vtk
from vtk.util import numpy_support as ns
import numpy as np

# input -> filename is a string
# input -> data is a vtk type (either ".vti" or ".vtp")
def WriteData(filename, data):
    writer = None
    extension = os.path.basename(filename).split('.')[-1]
    
    if extension == "vti":
        writer = vtk.vtkXMLImageDataWriter()
    elif extension == "vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        print("File Extension's Unknown")
        return
    
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Write()

# input filename is a string
def ReadVTK(filename):
    # figure out which file extension the filename has and read accordingly
    reader = None
    extension = os.path.basename(filename).split('.')[-1]
    
    if extension == "vti":
        reader = vtk.vtkXMLImageDataReader()
    elif extension == "vtp":
        reader = vtk.vtkXMLPolyDataReader()
    else:
        print("File Extension's Unknown")
        return
    
    reader.SetFileName(filename)
    reader.Update()
    
    return reader

# input -> bounds is a list
# image -> vtkImageData
def ConvertBoundsToExtents(image, bounds):
    bounds = np.array(bounds).astype(int).tolist()
    extents = [float()] * 6
    
    # compute the structured coordinates to find xmin, ymin, zmin
    pcoords = [float()] * 3
    x   = [bounds[0]-1, bounds[2]-1, bounds[4]-1]
    ijk = [bounds[1]+1, bounds[3]+1, bounds[5]+1]
    image.ComputeStructuredCoordinates(x, ijk, pcoords)
    
    extents[0] = ijk[0] + 1
    extents[2] = ijk[1] + 1
    extents[4] = ijk[2] + 1
    
    # compute the structured coordinates to find xmax, ymax, zmax
    pcoords = [float()] * 3
    x   = [bounds[0]-1, bounds[2]-1, bounds[4]-1]
    ijk = [bounds[1]+1, bounds[3]+1, bounds[5]+1]
    image.ComputeStructuredCoordinates(ijk, x, pcoords)
    
    extents[1] = x[0]
    extents[3] = x[1]
    extents[5] = x[2]
    
    # print out the conversion
    print("Conversion: {0}".format(extents))
    return extents

# surface -> vtkPolyData
# image -> vtkImageData
# tag_value -> int / float (up to the user)
def TagImageFromPolyDataSurface(surface, image, tag_value=1):
    # instantiate point locator in order to map data back to the image
    octree = vtk.vtkOctreePointLocator()
    octree.SetDataSet(image)
    octree.BuildLocator()
    
    # convert bounds of polydata to image extents
    extents = ConvertBoundsToExtents(image, surface.GetBounds())
    
    # extract a smaller in order to iterate over less points
    voi = vtk.vtkExtractVOI()
    voi.SetInputData(image)
    voi.SetVOI(extents)
    voi.Update()
    
    # find which points are within the surface (polydata) provided
    selection = vtk.vtkSelectEnclosedPoints()
    selection.SetSurfaceData(surface)
    selection.SetInputData(voi.GetOutput())
    selection.Update()
    
    # make sure to keep the same nomenclature for the vtk array to
    # not overwrite the current data
    scalars = ns.vtk_to_numpy(image.GetPointData().GetScalars())
    scalars_name = image.GetPointData().GetScalars().GetName()
    
    # iterate and find which points are within the volume
    for i in range(selection.GetOutput().GetNumberOfPoints()):
        if selection.IsInside(i) == True:
            iD = octree.FindClosestPoint(selection.GetOutput().GetPoint(i))
            scalars[iD] = tag_value
    
    # convert the numpy array to a vtk array
    s = ns.numpy_to_vtk(scalars)
    s.SetName(scalars_name)
    
    # update the image with the tagged values for the mapped volume
    image.GetPointData().SetScalars(s)

# using vtkProbeFilter, return the values along the line in order to map
# the data to a 1-Dimensional input for the neural network.
def InterpolatePoints(data, source):
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(data)
    probe.SetSourceData(source)
    probe.Update()
    
    return probe
