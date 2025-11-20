from nnwOverfittingresolution import NeuralNetwork
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

class trainer():
    def __init__(self):
        # Make local reference to Neural Network
        self.N = NeuralNetwork()

    def costFunctionWrapper(self, parms, x , y):
        self.N.setParams(parms)
        cost = self.N.costfunction(x , y)
        grad = self.N.computeGradients(x, y)
        return cost, grad

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costfunction(self.X, self.Y))
        self.testJ.append(self.N.costfunction(self.testX, self.testY))

    def train(self, trainX, trainY, testX, testY):
        # Make internal values for callback function
        self.X = trainX
        self.Y = trainY

        self.testX = testX
        self.testY = testY

        # Make list to store costs
        self.J = []
        self.testJ = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX , trainY), options=options, callback=self.callbackF)
        
        #Replace the random params with trained params
        self.N.setParams(_res.x)
        self.optimizationResults = _res

    
t = trainer()
t.train(t.N.trainX, t.N.trainY, t.N.testX, t.N.testY)

# Train and test batches
plt.subplot(1,2,1)
plt.scatter(t.N.trainX[:,0], t.N.trainY)
plt.grid(1)
plt.xlabel('Hours Sleeping')
plt.ylabel('Test Score')

plt.subplot(1,2,2)
plt.scatter(t.N.trainX[:,1], t.N.trainY)
plt.grid(1)
plt.xlabel('Hours Studying')
plt.ylabel('Test Score')
plt.show() 

#Plot cost during training:
plt.plot(t.J)
plt.plot(t.testJ)
plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show() 

#Test network for various combinations of sleep/study:
hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)

#Normalize data (same way training data way normalized)
hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.

#Create 2-d versions of input for plotting
a, b  = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

#Join into a single input matrix:
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()

allOutputs = t.N.forward(allInputs)

#Contour Plot:
yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

CS = plt.contour(xx, yy, 100 * allOutputs.reshape(100, 100))  # Use plt.contour
plt.clabel(CS, inline=1, fontsize=10)  # Use plt.clabel
plt.xlabel('Hours Sleep')  # Use plt.xlabel
plt.ylabel('Hours Study')  # Use plt.ylabel
plt.title('Predicted Test Scores Contour')  # Optional: Add a title
plt.show()

# 3D Plot of Training Data and Predictions
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm  # Add this import for colormap

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Fix: Use add_subplot instead of gca

# Scatter training examples (un-normalize X for original scale, scale Y back to 0-100)
ax.scatter(10 * t.N.trainX[:, 0], 5 * t.N.trainX[:, 1], 100 * t.N.trainY, c='k', alpha=1, s=30)

# Surface plot of predictions
surf = ax.plot_surface(xx, yy, 100 * allOutputs.reshape(100, 100), \
                       cmap=cm.jet, alpha=0.5)

ax.set_xlabel('Hours Sleep')
ax.set_ylabel('Hours Study')
ax.set_zlabel('Test Score')

plt.show()  # Display the 3D plot

# Test on a single value after training
single_input_raw = np.array([[11, 4]])  # Raw input (2D array)
train_max = np.array([10, 5])  # Max from training data (sleep:10, study:5)
single_input_norm = single_input_raw / train_max  # Normalize: [11/10, 4/5] = [1.1, 0.8]
single_true = 96

prediction = t.N.forward(single_input_norm) * 100  # Predict and un-normalize
print(f"Single Test Input: {single_input_norm}")
print(f"Predicted Score: {prediction[0][0]:.2f}%")
print(f"True Score: {single_true:.2f}%")
print(f"Error: {abs(prediction[0][0] - single_true):.2f}%")
