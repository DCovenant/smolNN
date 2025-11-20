# Simple Neural Network

To better understand what happens behind the curtains, a simple neural network was created from scratch.
Every line of code was learned by watching the series **Neural Network Demystified** by Welch Labs.
If you share the same interest, I highly recommend watching it:
https://www.youtube.com/playlist?list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU

## Project Overview

After learning the basics and understanding the ups and downs of neural networks, an experimental neural network was created.
The goal was to test if I could build something in scope but different from the tutorial, using the knowledge gained from studying the videos.

## The Iris Classification Problem

I chose to create a neural network for the **Iris dataset**. The Iris dataset contains 3 similar flower species (Setosa, Versicolor, and Virginica) that are identified by four features:
- Sepal length
- Sepal width
- Petal length
- Petal width 

## Network Architecture

To challenge myself beyond the tutorial, I designed a deeper network with 2 hidden layers (8 and 4 neurons respectively).
This architecture would force me to handle more complex gradient calculations and face real problems in implementation.

### Overall Structure:
- **Input layer**: 4 neurons (one for each feature)
- **Hidden layer 1**: 8 neurons
- **Hidden layer 2**: 4 neurons
- **Output layer**: 3 neurons (one for each class)

### Output Classes:
- Iris setosa: `[1, 0, 0]`
- Iris versicolor: `[0, 1, 0]`
- Iris virginica: `[0, 0, 1]`

*Note: The outputs are one-hot encoded, meaning only one value is 1 (the true class) and the rest are 0.*

### Activation Functions:
The base activation function is **sigmoid** (as used in the tutorial videos), with **softmax** added to the output layer for better multi-class classification.


## Mathematical Foundation

### Forward Propagation Equations

I started by writing on paper the mathematical functions that represent the neural network:

```
z2 = X·W1
a2 = f(z2)

z3 = a2·W2
a3 = f(z3)

z4 = a3·W3
yHat = f(z4)
```

Where:
- `X` = input data
- `W1, W2, W3` = weight matrices for each layer
- `z` = weighted sum (linear combination)
- `a` = activation (output after applying activation function)
- `f(·)` = activation function (sigmoid for hidden layers, softmax for output)
- `yHat` = predicted output

### Error Calculation

```
error: e = y - yHat
```
Where `y` is the desired output and `yHat` is the network's prediction.

### Cost Function (Mean Squared Error)

Since we're working with matrices (batches of data), the cost function is:

```
J = (1/2m) Σ (y - yHat)²
```

Where:
- `J` = total cost
- `m` = number of samples in the batch
- `Σ` = summation over all samples
- `(y - yHat)²` = squared error for each sample

**Why square the error?**
1. **Sign independence**: Whether the error is positive or negative, it contributes positively to the cost (errors don't cancel out)
2. **Penalty for large errors**: Squaring penalizes larger errors more than smaller ones, forcing the network to prioritize fixing big mistakes
3. **Mathematical convenience**: The squared error is smooth and differentiable, making gradient calculations straightforward

**Why divide by 2?**
The factor of 1/2 simplifies the derivative during backpropagation (the 2 from the power rule cancels with the 1/2).

**Why divide by m?**
Averaging over the batch size makes the cost independent of how many samples we process at once, ensuring consistent optimization.

## Backpropagation (Computing Gradients)

To train the network, we need to know **how each weight affects the cost**. This allows us to adjust weights in the direction that reduces the error.
We calculate this using **partial derivatives** of the cost with respect to each weight.

### Why Start from the End?
We start with the last layer (W3) because it directly connects to the output, making calculations easier. Then we work backwards through the network.

### Layer 3 Gradient (Output Layer)

Using the **chain rule**:
```
dJ/dW3 = (dJ/dyHat) · (dyHat/dz4) · (dz4/dW3)
```

Breaking it down:
- `dJ/dyHat = (yHat - y)` — how cost changes with prediction
- `dyHat/dz4 = f'(z4)` — derivative of activation function
- `dz4/dW3 = a3ᵀ` — derivative of weighted sum

Combining:
```
delta4 = (yHat - y) · f'(z4)
dJ/dW3 = a3ᵀ · delta4
```

### Layer 2 Gradient (Second Hidden Layer)

```
delta3 = (delta4 · W3ᵀ) · f'(z3)
dJ/dW2 = a2ᵀ · delta3
```

### Layer 1 Gradient (First Hidden Layer)

```
delta2 = (delta3 · W2ᵀ) · f'(z2)
dJ/dW1 = Xᵀ · delta2
```

**Note**: The transpose operation (ᵀ) ensures matrix dimensions align correctly for multiplication.

### Why Divide by Batch Size?

In the implementation, we divide each gradient by `m` (batch size):
```python
dJdW3 = np.dot(self.a3.T, delta4) / X.shape[0]
```
This matches the cost function's averaging, ensuring gradients represent the **mean gradient per sample** rather than the total sum.

With these three gradient functions, we have everything needed to train the neural network via gradient descent.

## Implementation

### Neural Network Class

```python
class IrisNeuralNetwork():
    def __init__(self):
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
        J = 0.5*np.sum((Y-yHat)**2)/X.shape[0]  
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

        return dJdW1, dJdW2, dJdW3
```

## Gradient Checking (Numerical Verification)

Before training, we verify that our analytical gradients (from backpropagation) are correct. We do this by computing gradients numerically using the finite difference method:

```
f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
```

This approximates the derivative by measuring how the cost changes with tiny weight perturbations (ε = 1e-4).

### Why is This Important?
- Backpropagation is complex and error-prone
- Small mistakes in gradient calculations lead to poor training
- Numerical gradients are slow but reliable—perfect for verification

### What to Expect:
The relative difference between analytical and numerical gradients should be **< 1e-8**. If it's larger, there's a bug in the backpropagation code.

```python
"""
Gradient descent check, should be a value < 1e-08
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
```

### Running the Gradient Check

```python
numgrad = nn.computeNumericalGradients(nn.trainX, nn.trainY)
grad = nn.computeGradients(nn.trainX, nn.trainY)

print("Gradient descent check:")
print(np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad))
# Expected output: ~1e-10 (very small number)
```

If the output is larger than 1e-8, there's an error in the gradient calculations that must be fixed before training.

---

# Training the Neural Network

## Training the Neural Network

To train the network, we need an optimization algorithm that will adjust the weights to minimize the cost function. We use **scipy.optimize.minimize** with the **BFGS** (Broyden–Fletcher–Goldfarb–Shanno) algorithm, which is a quasi-Newton method.

### Why BFGS?
- **Adaptive Learning**: Unlike basic gradient descent with a fixed learning rate, BFGS adapts the step size based on the curvature of the cost surface.
- **Faster Convergence**: It approximates the second derivatives (Hessian) to take more informed steps, converging much faster than simple gradient descent.
- **No Manual Tuning**: We don't need to manually set a learning rate—BFGS handles it automatically.

### The Trainer Class
The `trainer` class orchestrates the training process:

```python
class trainer():
    def __init__(self):
        self.nn = IrisNeuralNetwork()

    def costFunctionWrapper(self, params, X, Y):
        self.nn.setParams(params)
        cost = self.nn.costfunction(X, Y)
        grad = self.nn.computeGradients(X, Y)
        return cost, grad

    def callbackF(self, params):
        self.nn.setParams(params)
        self.J.append(self.nn.costfunction(self.X, self.Y))
        self.testJ.append(self.nn.costfunction(self.testX, self.testY))

    def train(self, trainX, trainY, testX, testY):
        self.X = trainX
        self.Y = trainY
        self.testX = testX
        self.testY = testY
        self.J = []
        self.testJ = []

        params0 = self.nn.getParams()
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, 
                                 method='BFGS', args=(trainX, trainY), 
                                 options=options, callback=self.callbackF)
        
        self.nn.setParams(_res.x)
        self.optimizationResults = _res
```

- **costFunctionWrapper**: Provides the cost and gradients to the optimizer in the format it expects.
- **callbackF**: Called after each iteration to record the training and test costs for monitoring.
- **train**: Sets up and runs the optimization, then stores the final trained weights.

### Training Progress
During training, the optimizer iteratively:
1. Evaluates the cost function on the training data
2. Computes the gradients via backpropagation
3. Updates the weights to reduce the cost
4. Repeats until convergence or max iterations

The callback function tracks both training cost (`self.J`) and test cost (`self.testJ`), allowing us to monitor for overfitting.

## Regularization (L2 Weight Decay)

To prevent overfitting, we add **L2 regularization** to the cost function:

```python
J = 0.5*np.sum((Y-yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
```

The term `(Lambda/2) * sum(weights^2)` penalizes large weight values:
- **Why?** Large weights can cause the network to fit noise in the training data (overfitting).
- **Effect**: Encourages smaller, smoother weights that generalize better to unseen data.
- **Lambda**: Controls the strength of regularization (0.00001 in our case—a small penalty).

The gradients also include the regularization term:
```python
dJdW3 = np.dot(self.a3.T, delta4) / X.shape[0] + self.Lambda * self.W3
```

## Understanding Overfitting

**Overfitting** occurs when a model learns the training data too well, including its noise and peculiarities, but fails to generalize to new, unseen data.

### Signs of Overfitting:
- **Low training cost**: The model fits the training data almost perfectly.
- **High test cost**: The model performs poorly on test data it hasn't seen.
- **Diverging costs**: Training cost decreases while test cost stays high or increases.

### Why Does Overfitting Happen?
1. **Too few training samples**: With only 9 training samples, the network can memorize them without learning general patterns.
2. **Too many parameters**: Our network has 76 weights (4→8→4→3). With more parameters than training samples, the model has enough capacity to "memorize" rather than "learn".
3. **Class imbalance**: If training data only contains examples from some classes, the network will fail on other classes.

### Example from Our Code:
When we trained on only 5 Setosa samples, the network learned to always predict Setosa with ~99.99% confidence, because that's all it had seen. When tested on Versicolor and Virginica samples, it failed completely—the test cost was very high (12.53).

### How to Combat Overfitting:
1. **More training data**: Use more samples with all classes represented (we now use 9 samples from all 3 classes).
2. **Regularization**: L2 penalty discourages large weights that cause overfitting.
3. **Simpler model**: Reduce the number of hidden neurons if the network is too complex.
4. **Early stopping**: Stop training when test cost starts increasing.
5. **Data augmentation**: Create variations of existing samples (less applicable to Iris).

### Monitoring Overfitting:
The `callbackF` function tracks both training and test costs. By plotting these over iterations:
- If both decrease together: Good generalization
- If training decreases but test stays high/increases: Overfitting

```python
plt.plot(t.J, label='Train Cost')
plt.plot(t.testJ, label='Test Cost')
plt.legend()
plt.show()
```

## Activation Functions

### Sigmoid
Used in hidden layers:
```python
sigmoid(x) = 1 / (1 + e^(-x))
```
- **Range**: (0, 1)
- **Use**: Squashes values to a probability-like range
- **Drawback**: Can cause vanishing gradients for very large/small inputs

### Softmax
Used in the output layer for multi-class classification:
```python
softmax(x) = exp(x_i) / sum(exp(x_j))
```
- **Why?** Converts raw outputs to probabilities that sum to 1
- **Benefit**: Better suited for multi-class problems than sigmoid
- **Output**: For Iris, produces [P(Setosa), P(Versicolor), P(Virginica)]

The predicted class is the one with the highest probability:
```python
predicted_class = np.argmax(y_pred, axis=1)
```

## Data Normalization

Before training, we normalize the input features to [0, 1]:
```python
train_max = np.amax(self.trainX, axis=0)
self.trainX = self.trainX / train_max
self.testX = self.testX / train_max
```

### Why Normalize?
- **Equal influence**: Features with different scales (e.g., sepal length: 4-8, petal width: 0.1-2.5) would otherwise dominate learning.
- **Faster convergence**: Neural networks train more efficiently with scaled inputs.
- **Numerical stability**: Prevents extreme values that could cause overflow/underflow.

### Key Point:
**Always use training statistics for test data**—using test data's max would be "data leakage" (cheating by using information from the test set).

## Visualizing Results

After training, we visualize predictions with bar plots showing:
- **Gray bars**: Predicted probabilities for all three classes
- **Blue bar**: True class (ground truth)
- **Red bar**: Predicted class (if different from true)

This makes it easy to see:
- How confident the network is (height of bars)
- Whether predictions are correct (blue matches the tallest bar)
- Common mistakes (red bar indicates misclassification)

## Results and Analysis

With 9 training samples (3 from each class):
- **Gradient check**: ~1e-10 (excellent—gradients are correct)
- **Training cost**: ~0.0002 (very low—good fit on training data)
- **Test cost**: Should be similar if the model generalizes well

### What Good Results Look Like:
- Predictions like `[0.95, 0.03, 0.02]` (confident and correct)
- Test cost close to training cost (good generalization)
- High accuracy on test samples

### What Overfitting Looks Like:
- Training cost near 0, but test cost high (like our initial 12.53)
- All predictions the same class (network memorized training data)
- No variation in predictions despite different inputs

## Conclusion

This neural network demonstrates the fundamentals of:
- Forward propagation (making predictions)
- Backpropagation (computing gradients)
- Optimization (adjusting weights to minimize error)
- Regularization (preventing overfitting)
- Multi-class classification (distinguishing between 3 Iris species)

The key lesson: **More representative training data is crucial**. Neural networks need diverse examples to learn patterns rather than memorize specific cases.

I also experimented adding a softmax and crossentropy to see the results, they were better but I didnt sudy them, the code is still functional.
