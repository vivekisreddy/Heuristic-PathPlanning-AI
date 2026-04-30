import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function for a given input.

    Parameters:
    - x (numpy.ndarray): Input values.

    Returns:
    - numpy.ndarray: Output values after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def predict(X, theta):
    """
    Make predictions using logistic regression.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - theta (numpy.ndarray): Coefficient vector.

    Returns:
    - numpy.ndarray: Predicted probabilities.
    """
    # Ensure theta is a numpy array
    theta = np.array(theta)

    # Calculate the linear combination of features and coefficients
    z = np.dot(X, theta)

    # Apply the sigmoid function to obtain probabilities
    probabilities = sigmoid(z)

    return probabilities

def cost_function(X, y, theta, regularization_param):
    """
    Compute the cost function for logistic regression.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - theta (numpy.ndarray): Coefficient vector.
    - regularization_param (float): Regularization parameter.

    Returns:
    - float: Cost function value.
    """
    m = len(y)
    h = predict(X, theta)
    regularization_term = (regularization_param / (2 * m)) * np.sum(theta[1:]**2)
    neg_log_likelihood = -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return neg_log_likelihood + regularization_term

def gradient_descent(X, y, theta, alpha, regularization_param, iterations, tolerance=1e-4):
    """
    Perform gradient descent to optimize logistic regression parameters.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - theta (numpy.ndarray): Coefficient vector.
    - alpha (float): Learning rate.
    - regularization_param (float): Regularization parameter.
    - iterations (int): Number of iterations.
    - tolerance (float): Convergence threshold.

    Returns:
    - numpy.ndarray: Optimized coefficient vector.
    """
    m = len(y)
    prev_cost = cost_function(X, y, theta, regularization_param)

    for iteration in range(iterations):
        h = predict(X, theta)
        gradient = (1 / m) * X.T @ (h - y)
        regularization_term = (regularization_param / m) * np.concatenate(([0], theta[1:]))
        theta = theta - alpha * (gradient + regularization_term)

        # Compute the cost function at the new parameters
        current_cost = cost_function(X, y, theta, regularization_param)

        # Check for convergence
        if abs(prev_cost - current_cost) < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break

        # Update the previous cost for the next iteration
        prev_cost = current_cost

    return theta