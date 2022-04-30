# Machine Learning Models From Scratch

There are two main goals for this repository. They are:
- Going deeper into the understanding of the popular machine learning models that are out there.
- Revising the underlying concept on how this models work.

> A man who truly understands can implement!

# Dependencies
Numpy is the most essential tool that will be used. To get it up and running: 

```
# if installed already on the machine. If ModuleNotFoundError comes up then you would have to intall it.
import numpy as np

#If not already installed, on your terminal, input the command below.
pip install numpy
```


I would be using the visual studio code IDE to carry out this project, although anyone of your choosing is most welcome.

The first model would be the **K-Nearest Neighbours** which can be found in the [K-Nearest Neighbors](./K_Nearest_Neighbors.py)
The K-Nearest Neighbor is best described with the familiar quote "Birds of the same feathers flock together".
It is a supervised machine learning algorithm that can be used for classification problems as well as regression problems.
Euclidean distances between test data points and training data points are calculated and the ones which are closest are used. A diagram explaining the concept is below!

-------
![KNN.jpg](https://docsdrive.com/images/ansinet/jas/2010/fig4-2k10-1841-1858.gif)

-------

The second model would be the **Simple Linear Regression** which can be found in the [Linear Regression](./linear_regression.py)
The Simple linear regression is an easy model. It is governed by the equation below:

![Linear Regression Image](https://jalammar.github.io/images/NNs_formula.png)


The model works because of the gradient descent algorithm which helps it updates the weights and biases that in turns helps the model to reduce the error. 


![Gradient Descent](https://www.oreilly.com/library/view/neural-networks-with/9781788397872/assets/56f6855e-0497-4a4e-8825-85c210e3420c.jpg)

------

This function is much like the linear regression function.
The difference is that the logistic regression is for classification problems. Usually used for the binary classification where there are only two possibilities of actions to choose from. 

It uses the linear function Y = Wx + b, but then goes on to be passed through the sigmoid activation function. After which gradient descent is performed to update the weigths and biases. 

The image below shows the distinction between the two functions (Linear regression and Logistic Regression)

![Logistic Regression vs Linear Regression](https://www.saedsayad.com/images/LogReg_1.png)

------

The next model is the **Naive Bayes** which can be found in the [Naive Bayes](./naive_bayes.py)
The formula employed in building it is below:

![Naive Bayes](https://miro.medium.com/max/1200/1*39U1Ln3tSdFqsfQy6ndxOA.png)
