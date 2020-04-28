from DefinedFunctions import *

class DataAcquisition():
    # source -> vtkImageData()
    # truth -> vtkImageData()
    
    def __init__(self, source_filename, truth_filename, debug=False):
        # source image filename
        self.sFilename = source_filename

        # truth image filename
        self.tFilename = truth_filename

        # instantiate reader / image information
        self.sReader = None
        self.tReader = None
        self.tImage = None
        self.sImage = None

        # input for neural network
        self.X = []
        
        # truth for neural network
        self.y = []
        
        # flag to either visualization data or not
        self.debug = debug
        
        # list to append polydatas to for debug
        self.append = None
        
        # line profile going through the volume
        self.line = vtk.vtkLineSource()
        
        # number of points along line profile to sample from
        self.numResolutionPoints = 0
        
        
    def AcquireData(self):
        # check for debug mode in order to save data for visualization
        if self.debug == True:
            self.append = vtk.vtkAppendPolyData()
        
        # read image files and get the image data output
        self.tReader = ReadVTK(self.sFilename)
        self.sReader = ReadVTK(self.tFilename)

        self.tImage = self.tReader.GetOutput()
        self.sImage = self.sReader.GetOutput()
        
        # iterate through the image to generate line profiles through
        # each point from the top of the image to the bottom... next
        # iteration of the file commit will include gathering lines
        # not only through each point, but a certain spacing between the
        # points to gather as much data as possible. I will probably
        # provide an argument that only selects the lines through the 
        # volume in order to analyze the profiles going through said
        # volumes (of interest).
        for i in range(self.sImage.GetNumberOfPoints()):
            if self.sImage.GetPoint(i)[2] > self.sImage.GetBounds()[5] - 1:
                current_point = self.sImage.GetPoint(i)
                
                # calculate number of Interpolation points (should ~ be 1 per
                # point data per line profile)
                self.numResolutionPoints = int((self.sImage.GetBounds()[5] - \
                    self.sImage.GetBounds()[4]) / self.sImage.GetSpacing()[2])
                
                # self line profile points and number of points to interpolate
                self.line.SetPoint1(current_point)
                self.line.SetPoint2(current_point[0],
                                    current_point[1],
                                    self.sImage.GetBounds()[4])
                self.line.SetResolution(self.numResolutionPoints)
                self.line.Update()
                
                # if debug, add line output to visualzation
                if self.debug == True:
                    append.AddInputData(self.line.GetOutput())
                
                # use a vtkProbeFilter
                sProbe = InterpolatePoints(self.line.GetOutput(), self.sImage)
                tProbe = InterpolatePoints(self.line.GetOutput(), self.tImage)
                
                # create empty lists to append necessary input information to
                intensity, truth_vals, z = [], [], []
                
                for j in range(self.numResolutionPoints):
                    intensity.append(sProbe.GetOutput().GetPointData().GetScalars().GetValue(j))
                    truth_vals.append(tProbe.GetOutput().GetPointData().GetScalars().GetValue(j))
                    z.append(sProbe.GetOutput().GetPoint(j)[2])
                
                # zip information together into a single array
                array = np.array(list(zip(intensity, truth_vals, z)))
                
                # sort the array based off the height of the interpolation point
                # of the line profile. 
                array = array[array[:, 2].argsort()]
                
                # iterate through the array and set up the input for the neural
                # network
                x = []
                for k in range(array.shape[0] - 1):
                    # utilizing the x-axis, turn the 1D data into 2D in order to use
                    # a direction vector mean as the weight
                    p1 = [k, array.T[0][k]]
                    p2 = [k+1, array.T[0][k+1]]
                    
                    # calculate the direction
                    direction = np.array(p2) - np.array(p1)
                    
                    # will be the second channel of the 1 Dimensional Neural Network
                    weight  = direction.mean()
                    
                    x.append([array.T[0][k], weight])
                
                self.X.append(x)
                self.y.append(array.T[1][:-1])
        
        # change the list input into a numpy array (it is the required type for
        # the neural network input
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)

        # if debug mode is true, then update the append pipeline, and write the
        # data to a polydata file.
        if self.debug == True:
            append.Update()
            WriteData("profiles.vtp", append.GetOutput())

