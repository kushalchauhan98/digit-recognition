 # digit-recognition
A demonstration of Neural Network and Support Vector Machine based classifier for digit recognition on the MNIST database.

Neural Network : A Neural Network with 2 hidden layers was used yeilding a classification accuracy of 96.5%.
Support Vector Machine : Support Vector Machine with Radial Basis Function Kernel was used to achieve a classification accuracy of 98%.

I have loaded a binary file "train" containing the following variables:

1.  X     - Training Set data
2.  y     - Training Labels
3.  Xval  - Cross Validation Set data
4.  yval  - Cross Validation Set Labels
5.  Xtest - Test Set data
6.  ytest - Test Set Labels

I haven't uploaded the "train" file for it is too big in size. You can create this file on your own using the MNIST data.

The number of hidden units in the hidden layers(in case of the Neural Network Classifier) and the values of gamma and C(in case of the Support Vector Machine Classifier) were decided after many runs of the program with different values, and the value which minimised the Cross Validation Set error was chosen.
