# LineProfileDeconvolution
uses line profiles to semantically segment an image in 1 Dimension

Note -> this is an initial release. The network requires more data (which will be coming soon) in order to get closer to the truth; as well as not overfit.

Data Acquisition from Line Profiles from the vertical direction through a volume
![Data Acquisition](https://github.com/kevindean/LineProfileDeconvolution/blob/master/Screenshot%20from%202020-04-25%2017-04-20.png)

Transcribe the data down to 1 Dimension
![Transcribe to Neural Net Input Data](https://github.com/kevindean/LineProfileDeconvolution/blob/master/LineProfileData.png)

Train the Network (see how to adjust the network structure / dataset to get the best results; the network is set up for semantic segmentation, represented as a unet-style architecture)
![Network Prediction of Line Profile in 1 Dimension](https://github.com/kevindean/LineProfileDeconvolution/blob/master/LineProfilePrediction.png)

Using a Point Locator, keep track of the indices and map the predictions back to a 3 Dimensional Volume
![Mapped back into a 3 Dimensional image as ---> Input, Prediction, Truth](https://github.com/kevindean/LineProfileDeconvolution/blob/master/Screenshot%20from%202020-04-25%2016-57-52.png)


2nd Note -> Analysis module will be coming soon.
