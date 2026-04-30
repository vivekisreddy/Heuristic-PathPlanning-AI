# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from logistic_regression import predict, gradient_descent
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values and converting the target variable to binary.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    # Handle missing values by replacing "?" with NaN
    df.replace('?', np.nan, inplace=True)

    # Handle remaining missing values by replacing NaN with mean
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)

    # Convert the target variable to binary (1 for 'ckd' and 0 for 'notckd')
    df['class'] = df['class'].apply(lambda check: 1 if check == 1 else 0 if check == 0 else np.nan)

    
    return df

from sklearn.model_selection import train_test_split

def split_data(X, y, train_ratio=0.8):
    """
    Split the data into training and testing sets using train_test_split.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.
    - train_ratio (float): Ratio of data to be used for training.

    Returns:
    - tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, shuffle=True, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def calculate_f_measure(y_true, y_pred):
    """
    Calculate precision, recall, and F-measure.

    Parameters:
    - y_true (numpy.ndarray): True labels.
    - y_pred (numpy.ndarray): Predicted labels.

    Returns:
    - float: F-measure.
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f_measure if not np.isnan(f_measure) else 0


def standardize_features(X):
    """
    Standardize features using Z-score standardization.

    Parameters:
    - X (pd.DataFrame): Feature matrix.

    Returns:
    - pd.DataFrame: Standardized feature matrix.
    """
    scaler = StandardScaler()
    standardized_X = scaler.fit_transform(X)
    return pd.DataFrame(standardized_X, columns=X.columns)


def main():
    # Load data and preprocess
    data = []
    with open('chronic_kidney_disease_full.arff', "r") as f:
        for line in f:
            line = line.replace('\n', '')
            data.append(line.split(','))
    column_names = ['age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                    'pvc', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'no_name', 'class']
    df = pd.DataFrame(data[145:], columns=column_names)
    df['class'] = df['class'].map({'ckd': 1, 'notckd': 0, '?': 'NaN'})
    df = preprocess_data(df)

    # Extract features and target variable
    X = df[['age', 'bgr']].copy()  # Adjust features as needed
    y = df['class']

    # Standardize features using Z-score standardization
    X_standardized = standardize_features(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_standardized, y)

    # Run logistic regression with different regularization parameters
    regularization_parameters = np.arange(-2.0, 4.2, 0.2)
    f_measures_train = []
    f_measures_test = []
    best_f_measure = 0

    for reg_param in regularization_parameters:
        # Initialize theta with different random values in each iteration
        np.random.seed(42)  # Set seed for reproducibility
        theta = np.random.rand(X_train.shape[1])

        # Run gradient descent
        alpha = 0.01
        iterations = 100
        theta = gradient_descent(X_train, y_train, theta, alpha, 10**reg_param, iterations)

        # Predictions on training set
        y_pred_train = predict(X_train, theta)
        f_measure_train = calculate_f_measure(y_train, (y_pred_train >= 0.5).astype(int))

        # Predictions on test set
        y_pred_test = predict(X_test, theta)
        f_measure_test = calculate_f_measure(y_test, (y_pred_test >= 0.5).astype(int))

        print(f"Reg. Param: {10**reg_param:.2e}, Train F-measure: {f_measure_train:.4f}, Test F-measure: {f_measure_test:.4f}")

        f_measures_train.append(f_measure_train)
        f_measures_test.append(f_measure_test)

        # Update the best F-measure if a higher value is obtained
        best_f_measure = max(best_f_measure, f_measure_test)

    # Plot the F-measure as a scatter plot
    plt.scatter(regularization_parameters, f_measures_train, label='Training Set', marker='o', color='blue')
    plt.scatter(regularization_parameters, f_measures_test, label='Test Set', marker='x', color='orange')
    plt.axvline(x=1.0, color='red', linestyle='--', label='Threshold for λ')
    plt.xlabel('Regularization Parameter (log scale)')
    plt.ylabel('F-measure')
    plt.title('F-measure vs Regularization Parameter')
    plt.legend()
    plt.show()

    return best_f_measure

if __name__ == "__main__":
    best_f_measure = main()
    print(f"Overall Best F-measure: {best_f_measure}")