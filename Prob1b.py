import pandas as pd
from sklearn.model_selection import train_test_split
from Prob1A import NaiveBayesClassifier

def calculate_f_measure(y_true, y_pred):
    """
    Calculate F-measure.

    Parameters:
    - y_true (list): True class labels.
    - y_pred (list): Predicted class labels.

    Returns:
    - float: F-measure.
    """
    # Count true positives, predicted positives, and actual positives
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    predicted_positives = sum(1 for pred in y_pred if pred == 1)
    actual_positives = sum(1 for true in y_true if true == 1)

    # Calculate precision, recall, and F-measure
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Return the calculated F-measure
    return f_measure

def main():
    # Load the datasets
    bodies_data = pd.read_csv('dbworld_bodies_stemmed.csv').drop(columns=['id'])
    subjects_data = pd.read_csv('dbworld_subjects_stemmed.csv').drop(columns=['id'])

    # Extract features (X) and labels (y) for email bodies
    X_bodies = bodies_data.iloc[:, :-1].values.tolist()
    y_bodies = bodies_data['CLASS'].tolist()

    # Extract features (X) and labels (y) for email subjects
    X_subjects = subjects_data.iloc[:, :-1].values.tolist()
    y_subjects = subjects_data['CLASS'].tolist()

    # Split the data into training and testing sets with stratification
    X_train_bodies, X_test_bodies, y_train_bodies, y_test_bodies = train_test_split(X_bodies, y_bodies, test_size=0.2, random_state=42, stratify=y_bodies, shuffle = True)
    X_train_subjects, X_test_subjects, y_train_subjects, y_test_subjects = train_test_split(X_subjects, y_subjects, test_size=0.2, random_state=42, stratify=y_subjects, shuffle = True)

    # Train and test the classifier for email bodies
    classifier_bodies = NaiveBayesClassifier(alpha=1)
    classifier_bodies.train(X_train_bodies, y_train_bodies)
    predictions_bodies = classifier_bodies.predict(X_test_bodies)
    f_measure_bodies = calculate_f_measure(y_test_bodies, predictions_bodies)

    # Train and test the classifier for email subjects
    classifier_subjects = NaiveBayesClassifier(alpha=1)
    classifier_subjects.train(X_train_subjects, y_train_subjects)
    predictions_subjects = classifier_subjects.predict(X_test_subjects)
    f_measure_subjects = calculate_f_measure(y_test_subjects, predictions_subjects)

    # Print the F-measures for both datasets
    print('Results')
    print('"dbworld_subjects_stemmed.csv" -----> F_Measure =', f_measure_subjects)
    print('"dbworld_bodies_stemmed.csv" -----> F_Measure =', f_measure_bodies)

if __name__ == "__main__":
    main()
