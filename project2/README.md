# Exploring the Transition from Regression Models to Neural Networks

Here we explore how neural networks and regression models (polynomial and logistic regression) perform on regression and classification learning tasks.

## Repository content

The repository contains the following files and folders:

- **LinearRegression.py** contains the class used for doing linear regression. When the ``lambda``-parameter is greater than zero, we do Ridge regression. 
The class has different solvers for finding optimal parameters, namely an analytic solution, stochastic gradient descent and gradient descent. 
The gradient descent methods can be applied with adaptive learning schemes, such as Adam, AdaGrad and RMSprop.
- **LogisticRegression.py** contains the class used to do logistic regression. 
Again, it is possible to do Ridge regression by setting ``lambda`` to a positve, non-zero value. 
Here we have implemented the stochastic gradient descent and gradient descent algorithms for finding optimal parameters.
Adaptive methods are available.
- **FFNN.py** holds the neural network class. The user can build a network with a flexible number of nodes and layers. 
Regularization of the weights and biases is also available. We implemented the stochastic gradient descent method for finding optimal model parameters.
- **Layer.py** is a class used for the layers in the neural network, storing the relevant variables of each layer. 
- **activation_functions.py** is just where we defined all the possible activation functions of the neural network.
- **GridSearch.py** contains basic plotting functions for doing grid searches of the various hyperparameters for the various models.
- **ResampleMethods.py** is were we have made functions for doing bootstrapping and k-fold cross validation.
- **common.py** has definitions of commonly used functions.
- **get_data.py** is a script used for generating the polynomial data set.
- **figs/** is a folder containing the figures produced.
- **datasets/** is a folder where the polynomial data is stored.

In addition, we have scripts used for testing the classes and functions, as well as scripts for doing the analyses:
- **test_FFNN_classification.py** is a script used to test the neural network on a classification task and compare it to the ``MLPCLassifier``-function from ``scikit-learn``.
- **test_FFNN_regression.py** tests the neural network on a regression task and compares it with the ``MLPRegressor``from ``scikit-learn``.
- **test_GridSearch.py** tests the GridSearch-functions.
- **test_LinearRegression.py** has a script for testing the linear regression class with different solvers.
- **test_LogisticRegression.py** tests the logistic regression class and compares it with ``SGDClassifier``from ``scikit-learn``
- **test_ResampleMethods.py** is used for applying and testing the resample methods on the various classes.
- **GDRegression_analysis.py** produces the results of polynomial regression solved with gradient descent.
- **Logistic_regression_analysis.py** produces results on logistic regression.
- **SGDRegression_analysis.py** produces results for polynomial regression solved with stochastic gradient descent.
- **FFNN_classification_analysis.py** produces results for the neural network when applied to the Wisconsin cancer data.
- **FFNN_regression_analysis.py** produces results for the neural network applied to the polynomial data set.



