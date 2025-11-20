"""
The expected ouputs can be:
Iris setosa	    [1, 0, 0]
Iris versicolor	[0, 1, 0]
Iris virginica	[0, 0, 1]  

The inputs are:
Sepal length
Sepal width
Petal length
Petal width
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy import optimize

class IrisNeuralNetwork():
    def __init__(self):
        # L2 Regularization/ Weight decay
        self.Lambda = 0.00001
        
        # Hyperparameters Definition
        self.inputLayerSize = 4
        self.hiddenLayer1Size = 8
        self.hiddenLayer2Size = 4
        self.outputLayersize = 3

        # Weights Definition
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayer1Size)
        self.W2 = np.random.randn(self.hiddenLayer1Size, self.hiddenLayer2Size)
        self.W3 = np.random.randn(self.hiddenLayer2Size, self.outputLayersize)

        # Training values
        self.trainX = [
            [5.1, 3.5, 1.4, 0.2],  # Setosa
            [4.9, 3.0, 1.4, 0.2],  # Setosa
            [4.7, 3.2, 1.3, 0.2],  # Setosa
            [7.0, 3.2, 4.7, 1.4],  # Versicolor
            [6.4, 3.2, 4.5, 1.5],  # Versicolor
            [6.9, 3.1, 4.9, 1.5],  # Versicolor
            [6.3, 3.3, 6.0, 2.5],  # Virginica
            [5.8, 2.7, 5.1, 1.9],  # Virginica
            [7.1, 3.0, 5.9, 2.1],  # Virginica
        ]
        self.trainY = [
            [1, 0, 0], [1, 0, 0], [1, 0, 0],  # Setosa
            [0, 1, 0], [0, 1, 0], [0, 1, 0],  # Versicolor
            [0, 0, 1], [0, 0, 1], [0, 0, 1],  # Virginica
        ]
        
        # Testing values
        self.testX = [
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 2.8, 4.9, 2.0],
            [6.3, 2.5, 4.9, 1.5],
            [5.7, 2.8, 4.1, 1.3],
            [6.7, 3.0, 5.0, 1.7]
        ]

        self.testY = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]

        # Normalizing data between 0 and 1
        train_max = np.amax(self.trainX, axis=0) # Compute BEFORE normalizing
        self.trainX = self.trainX/train_max
        self.testX = self.testX/train_max

    
    def sigmoid(self, x):
        """
        Sigmoid activation function
        """
        return 1/(1+np.exp(-x))

    def sigmoidPrime(self, x):
        """
        Derivative Sigmoid activation function
        """
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def softmax(self, x):
        """
        Since we are getting more than 2 classes has an output, 
        using softmax is a better alternative to sigmoidPrime
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation
        """
        self.z2 = np.dot(X ,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.softmax(self.z4)
        return yHat

    def costfunction(self, X, Y):
        """
        Cost functions, y - yhat (y -> value wanted, yHat -> prediction)
        """
        yHat = self.forward(X)
        # Cross-entropy loss
        J = -np.sum(Y * np.log(yHat + 1e-8)) / X.shape[0] + \
            (self.Lambda/2) * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
        return J

    def gradientDescent(self, X, Y):
        """
        Calculate how much a weight affects the cost
        """
        yHat = self.forward(X)
        
        # Delta calculation
        delta4 = (yHat - Y)
        delta3 = np.dot(delta4, self.W3.T) * self.sigmoidPrime(self.z3)
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        
        # Gradient Descent calc
        dJdW3 = np.dot(self.a3.T, delta4) / X.shape[0] + self.Lambda*self.W3
        dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] + self.Lambda*self.W2
        dJdW1 = np.dot(X.T, delta2) / X.shape[0] + self.Lambda*self.W1

        return dJdW1 ,dJdW2 ,dJdW3
    
    """
    Gradient descent check, should be a value > 1^-08
    """
    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
        return params
    
    def setParams(self,params):
        W1_start = 0
        W1_end = self.hiddenLayer1Size*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayer1Size))

        W2_end = W1_end+self.hiddenLayer1Size*self.hiddenLayer2Size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayer1Size, self.hiddenLayer2Size))

        W3_end = W2_end+self.outputLayersize*self.hiddenLayer2Size
        self.W3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayer2Size, self.outputLayersize))

    def computeGradients(self,X, Y):
        """
        Backpropagation
        """
        dJdW1, dJdW2, dJdW3 = self.gradientDescent(X, Y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel()))

    def computeNumericalGradients(self, X, Y):
        """
        Gradient descent check, actual calc
        """
        paramsInitial = self.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            perturb[p] = e
            self.setParams(paramsInitial + perturb)
            loss2 = self.costfunction(X, Y)

            self.setParams(paramsInitial - perturb)
            loss1 = self.costfunction(X, Y)

            numgrad[p] = (loss2-loss1) / (2*e)
            
            perturb[p] = 0

        self.setParams(paramsInitial)

        return numgrad

class trainer():
    def __init__(self):
        self.nn = IrisNeuralNetwork()

    def costFunctionWrapper(self, parms, X, Y):
        self.nn.setParams(parms)
        cost = self.nn.costfunction(X, Y)
        grad = self.nn.computeGradients(X, Y)
        return cost, grad

    def callbackF(self, params):
        self.nn.setParams(params)
        self.J.append(self.nn.costfunction(self.X, self.Y))
        self.testJ.append(self.nn.costfunction(self.testX, self.testY))

    def train(self, trainX, trainY, testX, testY):
        # Make internal values for callback function
        self.X = trainX
        self.Y = trainY

        self.testX = testX
        self.testY = testY

        # Make list to store costs
        self.J = []
        self.testJ = []

        params0 = self.nn.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX , trainY), options=options, callback=self.callbackF)
        
        #Replace the random params with trained params
        self.nn.setParams(_res.x)
        self.optimizationResults = _res

nn = IrisNeuralNetwork()

numgrad = nn.computeNumericalGradients(nn.trainX, nn.trainY)
grad = nn.computeGradients(nn.trainX, nn.trainY)

print("Gradient descent check:")
print(np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad))

t = trainer()
t.train(t.nn.trainX, t.nn.trainY, t.nn.testX, t.nn.testY)

print(t.nn.forward(t.nn.testX))
print(t.nn.costfunction(t.nn.testX,t.nn.testY))

# Get predictions for test set
y_pred = t.nn.forward(t.nn.testX)  # shape: (n_samples, 3)
y_true = np.array(t.nn.testY)      # shape: (n_samples, 3)
class_names = ['Setosa', 'Versicolor', 'Virginica']

num_samples = len(y_pred)
fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*3, 4), sharey=True)

x = np.arange(1, 4)  # 1, 2, 3 for the classes

for i in range(num_samples):
    ax = axes[i]
    # Plot all bars in gray
    bars = ax.bar(x, y_pred[i], color='gray', alpha=0.5)
    # Highlight true class in blue
    true_class = np.argmax(y_true[i])
    bars[true_class].set_color('blue')
    bars[true_class].set_alpha(0.7)
    # Highlight predicted class in red (if different from true)
    pred_class = np.argmax(y_pred[i])
    if pred_class != true_class:
        bars[pred_class].set_color('red')
        bars[pred_class].set_alpha(0.7)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_title(f'Test {i+1}')
    if i == 0:
        ax.set_ylabel('Predicted Probability')

plt.tight_layout()
plt.show()
