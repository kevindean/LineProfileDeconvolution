from DefinedFunctions import *

truth_reader = ReadVTK("truth.vti")
source_reader = ReadVTK("gaussian.vti")

truth_image = truth_reader.GetOutput()
source_image = source_reader.GetOutput()

octree = vtk.vtkOctreePointLocator()
octree.SetDataSet(source_image)
octree.BuildLocator()

X = []  # neural net input
y = []  # neural net truth

append = vtk.vtkAppendPolyData()

for i in range(source_image.GetNumberOfPoints()):
    if source_image.GetPoint(i)[2] > source_image.GetBounds()[5]-1:
        current_point = source_image.GetPoint(i)
        
        line = vtk.vtkLineSource()
        line.SetPoint1(current_point)
        line.SetPoint2(current_point[0], current_point[1], source_image.GetBounds()[4])
        
        num_resolution_points = int((source_image.GetBounds()[5] - source_image.GetBounds()[4]) / source_image.GetSpacing()[0])
        
        line.SetResolution(num_resolution_points)
        line.Update()
        
        append.AddInputData(line.GetOutput())
        
        source_probe = InterpolatePoints(line.GetOutput(), source_image)
        truth_probe = InterpolatePoints(line.GetOutput(), truth_image)
        
        intensity, truth_vals, z = [], [], []
        
        for j in range(num_resolution_points):
            intensity.append(source_probe.GetOutput().GetPointData().GetScalars().GetValue(j))
            truth_vals.append(truth_probe.GetOutput().GetPointData().GetScalars().GetValue(j))
            z.append(source_probe.GetOutput().GetPoint(j)[2])
        
        array = np.array(list(zip(intensity, truth_vals, z)))
        array = array[array[:,2].argsort()]
        
        x = []
        for k in range(array.shape[0] - 1):
            p1 = [k, array.T[0][k]]
            p2 = [k+1, array.T[0][k+1]]
            
            direction = np.array(p2) - np.array(p1)
            weight = direction.mean()
            
            x.append([array.T[0][k], weight])
        
        X.append(x)
        y.append(array.T[1][:-1])

X = np.asarray(X)
y = np.asarray(y).reshape(-1, num_resolution_points-1, 1)

append.Update()
WriteData("profiles.vtp", append.GetOutput())

# neural network
filters = 2
kernel = 2
pooling = 2
sampling = 2

inputs = Input(shape=(None, 2))

c1 = Conv1D(filters * 2, kernel, activation="relu", padding="same")(inputs)
c1 = Conv1D(filters * 2, kernel, activation="relu", padding="same")(c1)
m1 = MaxPooling1D(2)(c1)

middle = Conv1D(filters * 8, kernel, activation="relu", padding="same")(m1)
middle = Conv1D(filters * 16, kernel, activation="relu", padding="same")(middle)
middle = Conv1D(filters * 16, kernel, activation="relu", padding="same")(middle)
middle = Conv1D(filters * 8, kernel, activation="relu", padding="same")(middle)

up1 = UpSampling1D(2)(middle)
concat1 = concatenate([up1, c1])
c2 = Conv1D(filters * 2, kernel, activation="relu", padding="same")(concat1)
c2 = Conv1D(filters * 2, kernel, activation="relu", padding="same")(c2)

c3 = Conv1D(1, 1)(c2)

model = Model(inputs=inputs, outputs=c3)
model.compile(loss=Huber(), optimizer=Adam(lr=1e-4), metrics=['accuracy'])
model.summary()

callback_list = [ModelCheckpoint("best-model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')]

model.fit(X, y, validation_split=0.3, epochs=150, batch_size=32, callbacks=callback_list)

scalars = np.zeros(source_image.GetNumberOfPoints())
for i in range(source_image.GetNumberOfPoints()):
    if source_image.GetPoint(i)[2] > source_image.GetBounds()[5]-1:
        print(i)
        current_point = source_image.GetPoint(i)
        
        line = vtk.vtkLineSource()
        line.SetPoint1(current_point)
        line.SetPoint2(current_point[0], current_point[1], source_image.GetBounds()[4])
        
        num_resolution_points = int((source_image.GetBounds()[5] - source_image.GetBounds()[4]) / source_image.GetSpacing()[0])
        
        line.SetResolution(num_resolution_points)
        line.Update()
        
        source_probe = InterpolatePoints(line.GetOutput(), source_image)
        
        intensity, z, ids = [], [], []
        
        for j in range(num_resolution_points):
            intensity.append(source_probe.GetOutput().GetPointData().GetScalars().GetValue(j))
            z.append(source_probe.GetOutput().GetPoint(j)[2])
            ids.append(octree.FindClosestPoint(source_probe.GetOutput().GetPoint(j)))
        
        array = np.array(list(zip(intensity, z, ids)))
        array = array[array[:,1].argsort()]
        
        x = []
        for k in range(array.shape[0] - 1):
            p1 = [k, array.T[0][k]]
            p2 = [k+1, array.T[0][k+1]]
            
            direction = np.array(p2) - np.array(p1)
            weight = direction.mean()
            
            x.append([array.T[0][k], weight])
        
        x = np.asarray(x)
        y = array.T[1][:-1]
        ids = array.T[3][:-1].astype(int)
        
        prediction = model.predict(x.reshape(-1, x.shape[0], 2)).reshape(x.shape[0],)
        
        for l in range(len(ids)):
            scalars[ids[l]] = prediction[l]

pred_image = vtk.vtkImageData()
pred_image.DeepCopy(truth_image)

scalars = ns.numpy_to_vtk(scalars)
scalars.SetName("Prediction")

pred_image.GetPointData().SetScalars(scalars)
WriteData("prediction.vti", pred_image)
