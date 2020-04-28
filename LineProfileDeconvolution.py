from DefinedFunctions import *

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Input, LSTM, Conv1D, MaxPooling1D, UpSampling1D, concatenate, Dropout

# input_data and truth_data need to be numpy arrays (otherwise you will get an error from tensorflow/keras)
class LineProfileDeconvolution():
    
    # initialize variables
    def __init__(self, 
                 filters=1, 
                 kernels=1, 
                 pooling=1, 
                 sampling=1, 
                 callback_list=[ModelCheckpoint("best-model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')], 
                 loss=Huber(), 
                 optimizer=Adam(lr=1e-4),
                 metrics=['mse'],
                 input_data=None,
                 truth_data=None,
                 validation_split=True,
                 num_epochs=200,
                 batch_size=32,
                 debug=False):
        self.filters = filters
        self.kernels = kernels
        self.pooling = pooling
        self.sampling = sampling
        self.callback_list = callback_list
        self.model = None
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.input_data = input_data
        self.truth_data = truth_data
        self.validation_split = validation_split
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.debug = debug
        self.history = None
    
    # generate the model architecture
    def generate_model(self):
        inputs = Input(shape=(None, 2))

        # set up convolutional block
        c1 = Conv1D(self.filters * 2, self.kernels, activation="relu", padding="same")(inputs)
        c1 = Conv1D(self.filters * 2, self.kernels, activation="relu", padding="same")(c1)
        m1 = MaxPooling1D(self.pooling)(c1)
        
        middle = Conv1D(self.filters * 8, self.kernels, activation="relu", padding="same")(m1)
        middle = Conv1D(self.filters * 16, self.kernels, activation="relu", padding="same")(middle)
        middle = Conv1D(self.filters * 16, self.kernels, activation="relu", padding="same")(middle)
        middle = Conv1D(self.filters * 8, self.kernels, activation="relu", padding="same")(middle)
        
        # set up upsampling / deconvolutional portion of the network
        # utilizes a concatenate making it a convolutional network + recurrent
        up1 = UpSampling1D(self.sampling)(middle)
        concat1 = concatenate([up1, c1])
        c2 = Conv1D(self.filters * 2, self.kernels, activation="relu", padding="same")(concat1)
        c2 = Conv1D(self.filters * 2, self.kernels, activation="relu", padding="same")(c2)
        c3 = Conv1D(1, 1)(c2)
        
        # generate model based off input and output
        model = Model(inputs=inputs, outputs=c3)
        model.compile(loss=self.loss, 
            optimizer=self.optimizer, 
            metrics=self.metrics)
        
        if self.debug == True:
            model.summary()
        
        # assign the model to the model object
        self.model = model
        
    def TrainTheNetwork(self):
        #TODO: add an input that defines the validation set
        if self.validation_split == True:
            val_split = 0.3
        
            # fit the data (Train the network)
            self.history = self.model.fit(self.input_data,
                self.truth_data,
                validation_split = val_split,
                epochs = self.num_epochs,
                batch_size = self.batch_size,
                callbacks = self.callback_list)
        
        else:
            # fit the data (Train the network)
            self.history = self.model.fit(self.input_data,
                self.truth_data,
                epochs = self.num_epochs,
                batch_size = self.batch_size,
                callbacks = self.callback_list)


class PredictOnModel(LineProfileDeconvolution):
    def __init__(self, LineProfileDeconvolution, source_filename):
        # scalars for entering predicted values associated with the point data
        # of the 3D vtkImageData
        self.scalars = None
        
        # source image filename
        self.sFilename = source_filename
        
        # instantiate reader / image information
        self.sReader = None
        self.sImage = None
        
        # instantiate the point locator to map the values back to the image
        self.pLocator = vtk.vtkOctreePointLocator()
        
        # line profile going through the volume
        self.line = vtk.vtkLineSource()
        
        # number of points along line profile to sample from
        self.numResolutionPoints = 0

        # inherit from LineProfileDeconvolution class to get to the model
        # in order to generate a prediction per line profile
        self.lpd = LineProfileDeconvolution
        
        # set up object for predicted image
        self.pImage = None
    
    def PredictOnProfiles(self):
        self.sReader = ReadVTK(self.sFilename)
        self.sImage = self.sReader.GetOutput()
        
        # build the locator based off the image dataset
        self.pLocator.SetDataSet(self.sImage)
        self.pLocator.BuildLocator()
        
        # create an empty numpy array to store the predicted values for point
        self.scalars = np.zeros(self.sImage.GetNumberOfPoints())
        
        # iterate through the image, generate line profiles, and predict
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
                
                # use a vtkProbeFilter
                sProbe = InterpolatePoints(self.line.GetOutput(), self.sImage)
                
                # create empty lists to append necessary input information to
                intensity, z, ids = [], [], []
                
                for j in range(self.numResolutionPoints):
                    intensity.append(sProbe.GetOutput().GetPointData().GetScalars().GetValue(j))
                    z.append(sProbe.GetOutput().GetPoint(j)[2])
                    ids.append(self.pLocator.FindClosestPoint(sProbe.GetOutput().GetPoint(j)))
                
                # zip information together into a single array
                array = np.array(list(zip(intensity, z, ids)))
                
                # sort the array based off the height of the interpolation point
                # of the line profile. 
                array = array[array[:, 1].argsort()]
                
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
                
                # set up the input, truth, and point indices to map to
                x = np.asarray(x)
                ids = array.T[2][:-1].astype(int)

                # make a prediction on the line profile
                prediction = self.lpd.model.predict(x.reshape(-1, x.shape[0], 2)).reshape(x.shape[0],)
                
                # iterate through the array to map the prediction values to the correct point index
                # scalar
                for l in range(array.shape[0]-1):
                    self.scalars[ids[l]] = prediction[l]
            
            # set up image data by copying the input image
            # this assures that the origin, dimensions, extent, and spacing are all correct
            self.pImage = vtk.vtkImageData()
            self.pImage.DeepCopy(self.sImage)
            
            # use numpy support from vtk to turn the numpy array into a double array
            scalars = ns.numpy_to_vtk(self.scalars)
            scalars.SetName("Prediction")
            
            # overwrite the current scalar values with the predicted values.
            self.pImage.GetPointData().SetScalars(scalars)
            





