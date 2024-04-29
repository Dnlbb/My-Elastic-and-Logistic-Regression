# Custom Logistic Regression Models with MNIST

This project implements custom versions of Logistic Regression and Elastic-Net Logistic Regression in Python using NumPy, SciPy, and Pandas. The primary focus is on handling binary classification tasks using a subset of the MNIST dataset, specifically distinguishing between the digits 0 and 1.

## Features

- **Custom Logistic Regression**: A basic logistic regression model that predicts binary outcomes with batch gradient descent.
- **Custom Elastic-Net Logistic Regression**: Extends the basic model by incorporating L1 and L2 regularization to improve model generalization.
- **Batch Generator**: A utility function to generate batches of data, which is essential for batch gradient descent optimization.
- **Scikit-learn Compatibility**: Integration with Scikit-learn's pipeline and cross-validation tools to evaluate model performance.

## Dataset

The models are tested on a preprocessed version of the MNIST dataset, focusing on the digits 0 and 1. This subset allows us to treat the problem as a binary classification task.

## Usage

To run the logistic regression models:

1. Ensure that the MNIST dataset is correctly placed in your directory and properly named.
2. Set up a Python environment with the necessary packages installed (numpy, pandas, scipy, matplotlib, scikit-learn).
3. Run the script to train the models and evaluate them using cross-validation.

## Results

The models' performance is evaluated using 5-fold cross-validation. The mean accuracy provides a straightforward metric to gauge the effectiveness of the logistic regression implementations under regular conditions and with Elastic-Net regularization.

## Dependencies

- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn

## Files

- `logistic_regression.py`: Contains the implementation of the logistic regression models and the batch generator.
- `MNIST.csv`: The dataset file (ensure this is correctly named and located).

## Future Work

- Expand the model to handle multi-class classification.
- Implement additional optimizations for the training process.
- Enhance the batch generator for more complex batch strategies.

## Author

- Bykov Daniil
