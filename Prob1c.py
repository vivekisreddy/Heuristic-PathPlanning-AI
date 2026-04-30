from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import math

class NaiveBayesClassifier:
    def __init__(self, alpha=1):
        """
        Naive Bayes classifier with Laplacian smoothing.

        Parameters:
        - alpha (float): Laplacian smoothing parameter.
        """
        self.alpha = alpha
        self.class_probabilities = None
        self.feature_probabilities = None

    def train(self, X_train, y_train):
        """
        Train the Naive Bayes classifier.

        Parameters:
        - X_train (list of lists): List of training documents, where each document is represented as a list of words.
        - y_train (list): List of corresponding class labels.
        """
        total_documents = len(X_train)
        class_counts = {}
        feature_counts = {}

        for doc, label in zip(X_train, y_train):
            if label not in class_counts:
                class_counts[label] = 1
                feature_counts[label] = {}
            else:
                class_counts[label] += 1

            for word in doc:
                if word not in feature_counts[label]:
                    feature_counts[label][word] = 1
                else:
                    feature_counts[label][word] += 1

        self.class_probabilities = {label: count / total_documents for label, count in class_counts.items()}
        total_features = len(set(feature for doc in X_train for feature in doc))
        self.feature_probabilities = {label: {word: (count + self.alpha) / (sum(feature_counts[label].values()) + self.alpha * total_features) for word, count in feature_count.items()} for label, feature_count in feature_counts.items()}

    def predict(self, X_test):
        """
        Predict the class labels for a list of test documents.

        Parameters:
        - X_test (list of lists): List of test documents, where each document is represented as a list of words.

        Returns:
        - list: Predicted class labels for each test document.
        """
        predictions = []
        for doc in X_test:
            scores = {}
            for label, class_prob in self.class_probabilities.items():
                score = sum([math.log(class_prob)] + [math.log(self.feature_probabilities[label].get(word, self.alpha / (sum(self.feature_probabilities[label].values()) + self.alpha * len(self.feature_probabilities[label])))) for word in doc])
                scores[label] = score

            predicted_label = max(scores, key=scores.get)
            predictions.append(predicted_label)

        return predictions

def calculate_f_measure(y_true, y_pred):
    f_measure = f1_score(y_true, y_pred, pos_label=1, average='binary')
    return f_measure

def train_and_evaluate_sklearn(X_train, y_train, X_test, y_test):
    # Use LabelEncoder for mapping class labels
    label_encoder = LabelEncoder()
    y_train_mapped = label_encoder.fit_transform(y_train)
    y_test_mapped = label_encoder.transform(y_test)

    # Train the classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train_mapped)

    # Predict using the trained classifier
    predictions = clf.predict(X_test)

    # Calculate F-measure
    f_measure = calculate_f_measure(y_test_mapped, predictions)

    return f_measure

def main():
    # Load the datasets
    bodies_data = pd.read_csv('dbworld_bodies_stemmed.csv')
    subjects_data = pd.read_csv('dbworld_subjects_stemmed.csv')

    # Drop the 'id' column
    bodies_data = bodies_data.drop(columns=['id'])
    subjects_data = subjects_data.drop(columns=['id'])

    # Extract features (X) and labels (y)
    X_bodies = bodies_data.iloc[:, :-1].values
    y_bodies = bodies_data['CLASS'].values

    X_subjects = subjects_data.iloc[:, :-1].values
    y_subjects = subjects_data['CLASS'].values

    # Use StratifiedShuffleSplit for splitting the data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Train and evaluate scikit-learn classifier for email bodies
    for train_index, test_index in sss.split(X_bodies, y_bodies):
        X_train_bodies, X_test_bodies = X_bodies[train_index], X_bodies[test_index]
        y_train_bodies, y_test_bodies = y_bodies[train_index], y_bodies[test_index]

    f_measure_bodies = train_and_evaluate_sklearn(X_train_bodies, y_train_bodies, X_test_bodies, y_test_bodies)

    # Train and evaluate scikit-learn classifier for email subjects
    for train_index, test_index in sss.split(X_subjects, y_subjects):
        X_train_subjects, X_test_subjects = X_subjects[train_index], X_subjects[test_index]
        y_train_subjects, y_test_subjects = y_subjects[train_index], y_subjects[test_index]

    f_measure_subjects = train_and_evaluate_sklearn(X_train_subjects, y_train_subjects, X_test_subjects, y_test_subjects)

    # Print the F-measures for both datasets
    print('Results')
    print('"dbworld_subjects_stemmed.csv" -----> F_Measure =', f_measure_subjects)
    print('"dbworld_bodies_stemmed.csv" -----> F_Measure =', f_measure_bodies)

if __name__ == "__main__":
    main()
