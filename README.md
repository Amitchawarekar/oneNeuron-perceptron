# oneNeuron-perceptron

## Add image 

![sample image](plots/and.png)


<img src="plots/and.png" alt="and" width="500" height="600">

## Python Code

``` python
def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4 # SMALL WEIGHT INIT
    print(f"initial weights before training: \n{self.weights}")
    self.eta = eta # LEARNING RATE
    self.epochs = epochs 


  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights) # z = W * X
    return np.where(z > 0, 1, 0) # CONDITION, IF TRUE, ELSE
```

## Dataset

x1 | x2 | y
-|-|-
0|0|0
0|1|0
1|0|0
1|1|1

###

* point 1
* point 2


      
    
