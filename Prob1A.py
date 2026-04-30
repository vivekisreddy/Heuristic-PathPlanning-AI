from collections import defaultdict
import math


class NaiveBayesClassifier:
    def __init__(self, alpha=1):
        """
        Naive Bayes classifier with Laplacian smoothing.

        Parameters:
        - alpha (float): Laplacian smoothing parameter.
        """
        # Initialize the NaiveBayesClassifier with a specified alpha value
        self.alpha = alpha
        # Dictionary to store class probabilities
        self.class_probabilities = defaultdict(float)
        # Nested dictionary to store feature probabilities with Laplacian smoothing
        self.feature_probabilities = defaultdict(lambda: defaultdict(lambda: self.alpha))

    def train(self, X_train, y_train):
        """
        Train the Naive Bayes classifier.

        Parameters:
        - X_train (list of lists): List of training documents, where each document is represented as a list of words.
        - y_train (list): List of corresponding class labels.
        """
        # Calculate the total number of training documents
        total_documents = len(X_train)
        # Dictionary to store the count of each class
        class_counts = defaultdict(int)
        # Nested dictionary to store the count of each feature for each class
        feature_counts = defaultdict(lambda: defaultdict(int))

        # Combine X_train and y_train
        combined_data = list(zip(X_train, y_train))

        # Iterate through training data to calculate class and feature counts
        for doc, label in combined_data:
            # Increment the count of the current class
            class_counts[label] += 1
            # Iterate through words in the document and update feature counts
            for i, word in enumerate(doc):
                # Use the index 'i' as the key for feature counts
                feature_counts[label][i] += word

        # Calculate class probabilities based on the count of each class
        for label, count in class_counts.items():
            self.class_probabilities[label] = count / total_documents

        # Calculate the total number of unique features across all documents
        total_features = len(X_train[0])

        # Calculate feature probabilities with Laplacian smoothing
        for label, feature_count in feature_counts.items():
            total_words_in_class = sum(feature_count.values())
            for word, count in feature_count.items():
                self.feature_probabilities[label][word] = (count + self.alpha) / (total_words_in_class + self.alpha * total_features)

    def predict(self, X_test):
        """
        Predict the class labels for a list of test documents.

        Parameters:
        - X_test (list of lists): List of test documents, where each document is represented as a list of words.

        Returns:
        - list: Predicted class labels for each test document.
        """
        # List to store predicted class labels
        predictions = []
        # Iterate through test documents
        for doc in X_test:
            # Dictionary to store scores for each class
            scores = defaultdict(float)
            # Iterate through classes and calculate scores
            for label, class_prob in self.class_probabilities.items():
                # Initialize score with the log of class probability
                score = math.log(class_prob)
                # Iterate through words in the document and update scores
                for i, word in enumerate(doc):
                    if word > 0:
                        # Use the index 'i' for accessing feature probabilities
                        # Update the score only for words present in the document
                        score += math.log(self.feature_probabilities[label][i])

                # Store the final score for the class
                scores[label] = score

            # Predict the class label with the highest score
            predicted_label = max(scores, key=scores.get)
            # Append the predicted label to the list of predictions
            predictions.append(predicted_label)

        # Return the list of predicted class labels
        return predictions