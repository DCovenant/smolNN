import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

class NeuralNetwork:
    def __init__(self):
        # Regularization Parameter
        self.Lambda = 0.0001

        # Define HyperParameters
        self.inputlayersize=2
        self.hiddenlayersize=3
        self.outputlayersize=1

        # Weights Params
        self.W1 = np.random.randn(self.inputlayersize, self.hiddenlayersize)
        self.W2 = np.random.randn(self.hiddenlayersize, self.outputlayersize)

        # Training data (Hours sleeping / hours studying)
        trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
        trainY = np.array(([75], [82], [93], [70]), dtype=float) # test score

        # Test data (real world simulation)
        testX = np.array([[4, 5.5], [3,6],[9,9],[6,10]], dtype=float)
        testY = np.array([[55],[91],[23],[89]], dtype=float)

        # Normalizing data between 0 and 1
        self.trainX = trainX/np.amax(trainX, axis=0)
        self.trainY = trainY/100. #the max test score is 100

        self.testX = testX/np.amax(trainX, axis=0)
        self.testY = testY/100.
                

    """
    Sigmoid activation function
    """
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    """
    Derivative Sigmoid activation function
    """
    def sigmoidPrime(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    """
    Forward propagation
    """
    def forward(self, x):
        self.z2 = np.dot(x ,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def costfunction(self, x, y ):
        self.yHat = self.forward(x)
        J = 0.5*np.sum((y-self.yHat)**2)/x.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))  
        return J 

    def gradientDescent(self,x,y):
        """Cost function prime"""
        self.yHat = self.forward(x)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        # self.a2.T * delta3 -> this function represents the slope of across all of the values of the matrix
        # WE ARE WORKING WITH MATRIXES, dont forget
        # To have the average of all of the values we must sum them all and devide by the total amout of values.   
        # np.dot(self.a2.T,delta3)/x.shape[0] 
        dJdW2 = np.dot(self.a2.T,delta3)/x.shape[0] + self.Lambda*self.W2 # L2 regularization/Weight decay
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(x.T,delta2)/x.shape[0] + self.Lambda*self.W1

        return dJdW1, dJdW2
    
    def getParams(self):
        params = np.concatenate((self.W1.ravel(),self.W2.ravel()))
        return params

    def setParams(self,params):
        W1_start = 0
        W1_end = self.hiddenlayersize*self.inputlayersize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputlayersize, self.hiddenlayersize))

        W2_end = W1_end+self.outputlayersize*self.hiddenlayersize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenlayersize, self.outputlayersize))

    def computeGradients(self,x, y):
        """backpropagation"""
        dJdW1, dJdW2 = self.gradientDescent(x,y)
        return np.concatenate((dJdW1.ravel(),dJdW2.ravel()))
    
    def computeNumericalGradients(self,x, y):
        """gradient descent check"""
        paramsInitial = self.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            perturb[p] = e
            self.setParams(paramsInitial + perturb)
            loss2 = self.costfunction(x, y)

            self.setParams(paramsInitial - perturb)
            loss1 = self.costfunction(x, y)

            numgrad[p] = (loss2-loss1) / (2*e)
            
            perturb[p] = 0

        self.setParams(paramsInitial)

        return numgrad

nn = NeuralNetwork()

numgrad = nn.computeNumericalGradients(nn.trainX, nn.trainY)
grad = nn.computeGradients(nn.trainX, nn.trainY)

print("Gradient descent check:")
print(np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)) # should be under 1e-08