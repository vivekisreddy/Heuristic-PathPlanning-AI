import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Sigmoid function for logistic regression.

    Parameters:
    - x (numpy.ndarray): Input to the sigmoid function.

    Returns:
    - numpy.ndarray: Output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def cost_function(X, y, theta):
    """
    Cost function for logistic regression with regularization.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - theta (numpy.ndarray): Model parameters.
    - lambda_reg (float): Regularization parameter.

    Returns:
    - float: Cost value.
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    loss = -1/m * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return loss


    theta = np.zeros(X.shape[1])
    loss = cost_function(theta, X, y)
    print("Logistic Loss:", loss)

def gradient_descent(X, y, theta, alpha, lambda_reg, iterations):
    """
    Gradient descent optimization algorithm.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - theta (numpy.ndarray): Model parameters.
    - alpha (float): Learning rate.
    - lambda_reg (float): Regularization parameter.
    - iterations (int): Number of iterations.

    Returns:
    - tuple: Tuple containing updated parameters and history of cost values.
    """
    m = len(y)
    J_history = np.zeros(iterations)

    for i in range(iterations):
        h = sigmoid(X @ theta)
        errors = h - y
        reg_term = (lambda_reg / m) * theta[1:]

        # Update theta parameters
        theta[0] = theta[0] - (alpha / m) * np.sum(errors)
        theta[1:] = theta[1:] - (alpha / m) * (X[:, 1:].T @ errors + reg_term)

        # Calculate and store the cost function value
        J_history[i] = cost_function(X, y, theta, lambda_reg)

    return theta, J_history

def predict(X, theta, threshold=0.5):
    """
    Prediction function using the learned parameters.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - theta (numpy.ndarray): Model parameters.
    - threshold (float): Threshold for classification.

    Returns:
    - numpy.ndarray: Binary predictions.
    """
    return sigmoid(X @ theta) >= threshold

def calculate_f_measure(true_positives, false_positives, false_negatives):
    """
    Calculate F-measure given true positives, false positives, and false negatives.

    Parameters:
    - true_positives (int): Number of true positives.
    - false_positives (int): Number of false positives.
    - false_negatives (int): Number of false negatives.

    Returns:
    - float: F-measure.
    """
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f_measure = (2 * precision * recall) / (precision + recall + 1e-8)
    return f_measure

def split_train_test_data(X, y, train_ratio=0.8, random_seed=None):
    """
    Split the data into training and testing sets.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - train_ratio (float): Ratio of data used for training.
    - random_seed (int): Seed for reproducibility.

    Returns:
    - tuple: Tuple containing training and testing sets for features and target variable.
    """
    m = len(y)
    train_size = int(m * train_ratio)
    indices = np.arange(m)
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    X_train, X_test = X[train_indices, :], X[test_indices, :]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def standardize_features(X_train, X_test):
    """
    Normalize features using standardization.

    Parameters:
    - X_train (numpy.ndarray): Training set features.
    - X_test (numpy.ndarray): Testing set features.

    Returns:
    - tuple: Tuple containing standardized training and testing sets for features.
    """
    mean = np.mean(X_train, axis=0)
    std_dev = np.std(X_train, axis=0)
    X_train_std = (X_train - mean) / std_dev
    X_test_std = (X_test - mean) / std_dev
    return X_train_std, X_test_std

def add_intercept_term(X):
    """
    Add an intercept term to the feature matrix.

    Parameters:
    - X (numpy.ndarray): Feature matrix.

    Returns:
    - numpy.ndarray: Feature matrix with an added intercept term.
    """
    return np.c_[np.ones(X.shape[0]), X]

def logistic_regression(X, y, alpha, lambda_reg, iterations):
    """
    Logistic regression function with training and evaluation.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - alpha (float): Learning rate.
    - lambda_reg (float): Regularization parameter.
    - iterations (int): Number of iterations.

    Returns:
    - numpy.ndarray: Learned parameters.
    """
    X = add_intercept_term(X)
    theta = np.zeros(X.shape[1])
    
    theta, _ = gradient_descent(X, y, theta, alpha, lambda_reg, iterations)

    return theta

def plot_f_measure(lambda_values, f_measure_train, f_measure_test):
    """
    Plot F-measure vs. Lambda values.

    Parameters:
    - lambda_values (numpy.ndarray): Array of lambda (regularization parameter) values.
    - f_measure_train (numpy.ndarray): F-measure values for the training set.
    - f_measure_test (numpy.ndarray): F-measure values for the test set.
    """
    plt.plot(lambda_values, f_measure_train, label='Training Set')
    plt.plot(lambda_values, f_measure_test, label='Test Set')
    plt.xlabel('Lambda (Regularization Parameter)')
    plt.ylabel('F-Measure')
    plt.title('F-Measure vs. Lambda')
    plt.legend()
    plt.show()

def preprocess_data(X):
    """
    Preprocess the input data.

    Parameters:
    - X (pd.DataFrame): Input data.

    Returns:
    - numpy.ndarray: Preprocessed feature matrix.
    """
    # Handle missing values by replacing "?" with NaN
    X.replace('?', np.nan, inplace=True)

    # Handle remaining missing values by replacing NaN with mean
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)

    return X

def calculate_f_measure(y_true, y_pred):
    """
    Calculate F-measure given true and predicted labels.

    Parameters:
    - y_true (numpy.ndarray): True labels.
    - y_pred (numpy.ndarray): Predicted labels.

    Returns:
    - float: F-measure.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_measure = (2 * precision * recall) / (precision + recall + 1e-8)

    return f_measure

def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    """
    Evaluate a classifier on training and test sets.

    Parameters:
    - classifier: Classifier object with fit and predict methods.
    - X_train (numpy.ndarray): Training set features.
    - X_test (numpy.ndarray): Testing set features.
    - y_train (numpy.ndarray): Training set labels.
    - y_test (numpy.ndarray): Testing set labels.

    Returns:
    - tuple: Tuple containing F-measure values for training and test sets.
    """
    # Train the classifier
    classifier.fit(X_train, y_train)

    # Predict on training and test sets
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    # Evaluate F-measure
    f_measure_train = calculate_f_measure(y_train, y_train_pred)
    f_measure_test = calculate_f_measure(y_test, y_test_pred)

    return f_measure_train, f_measure_test

def svm_linear_classifier(X_train, X_test, y_train, y_test):
    """
    Support Vector Machine with linear kernel and default parameters.

    Parameters:
    - X_train (numpy.ndarray): Training set features.
    - X_test (numpy.ndarray): Testing set features.
    - y_train (numpy.ndarray): Training set labels.
    - y_test (numpy.ndarray): Testing set labels.

    Returns:
    - tuple: Tuple containing F-measure values for training and test sets.
    """
    svm_linear = SVC(kernel='linear')
    f_measure_train, f_measure_test = evaluate_classifier(svm_linear, X_train, X_test, y_train, y_test)
    return f_measure_train, f_measure_test

def svm_rbf_classifier(X_train, X_test, y_train, y_test):
    """
    Support Vector Machine with RBF kernel and default parameters.

    Parameters:
    - X_train (numpy.ndarray): Training set features.
    - X_test (numpy.ndarray): Testing set features.
    - y_train (numpy.ndarray): Training set labels.
    - y_test (numpy.ndarray): Testing set labels.

    Returns:
    - tuple: Tuple containing F-measure values for training and test sets.
    """
    svm_rbf = SVC(kernel='rbf')
    f_measure_train, f_measure_test = evaluate_classifier(svm_rbf, X_train, X_test, y_train, y_test)
    return f_measure_train, f_measure_test

def random_forest_classifier(X_train, X_test, y_train, y_test):
    """
    Random Forest with default parameters.

    Parameters:
    - X_train (numpy.ndarray): Training set features.
    - X_test (numpy.ndarray): Testing set features.
    - y_train (numpy.ndarray): Training set labels.
    - y_test (numpy.ndarray): Testing set labels.

    Returns:
    - tuple: Tuple containing F-measure values for training and test sets.
    """
    rf_classifier = RandomForestClassifier()
    f_measure_train, f_measure_test = evaluate_classifier(rf_classifier, X_train, X_test, y_train, y_test)
    return f_measure_train, f_measure_test

def main():
    """
    Main function to execute the support vector machine and random forest classifiers on the provided dataset
    and evaluate their performance using F-measure.
    """
    # Load the data from chronic_kidney_disease_full.arff file
    data = []
    with open('chronic_kidney_disease_full.arff', "r") as f: 
        for line in f: 
            line = line.replace('\n', '')
            data.append(line.split(','))
    column_names = ['age', 'bp', 'al', 'su', 'rbc', 'pc','pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pvc', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'no_name', 'class']
    df = pd.DataFrame(data[145:], columns=column_names)

    # Map 'ckd' and 'notckd' to 1 and 0 for the 'class' attribute
    df['class'] = df['class'].map({'ckd': 1, 'notckd': 0, '?': 'NaN'})

    # Preprocess and encode the data
    X = preprocess_data(df)
    X = X.dropna(how='all', axis=0)
    y = X['class'].values
    X = X.drop('class', axis=1).values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test_data(X, y)

    # Support Vector Machine with linear kernel
    svm_linear = SVC(kernel='linear')
    f_measure_train_svm_linear, f_measure_test_svm_linear = evaluate_classifier(svm_linear, X_train, X_test, y_train, y_test)

    # Support Vector Machine with RBF kernel
    svm_rbf = SVC(kernel='rbf')
    f_measure_train_svm_rbf, f_measure_test_svm_rbf = evaluate_classifier(svm_rbf, X_train, X_test, y_train, y_test)

    # Random Forest
    rf_classifier = RandomForestClassifier()
    f_measure_train_svm_linear, f_measure_test_svm_linear = svm_linear_classifier(X_train, X_test, y_train, y_test)
    print("Support Vector Machine with Linear Kernel:")
    print("F-measure on training set:", f_measure_train_svm_linear)
    print("F-measure on test set:", f_measure_test_svm_linear)

    # Support Vector Machine with RBF Kernel
    f_measure_train_svm_rbf, f_measure_test_svm_rbf = svm_rbf_classifier(X_train, X_test, y_train, y_test)
    print("\nSupport Vector Machine with RBF Kernel:")
    print("F-measure on training set:", f_measure_train_svm_rbf)
    print("F-measure on test set:", f_measure_test_svm_rbf)

    # Random Forest
    f_measure_train_rf, f_measure_test_rf = random_forest_classifier(X_train, X_test, y_train, y_test)
    print("\nRandom Forest:")
    print("F-measure on training set:", f_measure_train_rf)
    print("F-measure on test set:", f_measure_test_rf)

if __name__ == "__main__":
    main()
