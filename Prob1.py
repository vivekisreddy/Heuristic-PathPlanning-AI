# Naive Bayes Algorithm
# Train MultinomialNB(C,D)
# v = ExtractVocabulary(D)
# N = CountDocs(D)
# for each c in C
#     do Nc = CountDocsInClass(D,c)
#     prior[c] = Nc/N
#     textc = ConcatenateTextOfAllDocsInClass(D,c)
#     for each t in v 
#     do Tct = CountTokensOfTerm(textc, t)
#     for each t in v 
#     do condprob[t][c] = (Tct +1)/(sumt'(Tct' + 1))
# return V, prior, condprob

# ApplyMultinomialNB(C,V,prior, condprop, d)
# W = ExtractTokensFromDoc(V,d)
# for each c in C
# do score[c] = log prior[c]
#     for each t in W
#     do score[c]+=log condprop[t][c]
# return arg maxcinCscore[c]

import pandas as pd
class Main:

    bodies_data = pd.read_csv("dbworld_bodies_stemmed.csv", header = None) #puts the bodies data in to a dataframe to access easier
    subjects_data = pd.read_csv("dbworld_bodies_stemmed.csv", header = None) #puts the bodies data into a dataframe to access easier 


    bodies_data = bodies_data.drop(columns=0) #removes the id column
    subjects_data = subjects_data.drop(columns=0)


    body_features = bodies_data.iloc[:, :-1]  # Separates features (words) for both datasets 
    subject_features = subjects_data.iloc[:, :-1]  

    body_labels = bodies_data.iloc[:, -1]  # Separates labels (CLASS) - if the email should be discarded or not
    subject_labels = subjects_data.iloc[:, -1] 


    def fit(self, X, y):  # X is a np nd array where the num of rows is the num of samples and the num of columns is the num of features
    # y is a 1d row vector also of size n_samples of the labels
        #number of samples and number of features
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # will find the unique elements of an array, so if you have 2 classes, 0 and 1 it will find 
        n_classes = len(self._classes)
        self._mean = np.zeroes((n_classes, n_features), dtype = np.float64) # for each class (0 or 1) we need the mean for each feature
        self._var = np.zeroes((n_classes, n_features), dtype = np.float64) # for each class (0 or 1) we need the variance for each feature
        self._priors = np.zeroes(n_classes, dtype = np.float64) # for each class (0 or 1) we need the priors

        #for each class in classes 
        #finds the mean, var, and priors for each feature (word)
        for c in self.classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples) #gets the num of samples with this label and divides it by the num of total samples

    
    #for multiple samples
    def predict(self, X):
        y_pred = [self._predict(x) for x in X] #does this for each sample in the iist of test samples
        return y_pred
    
    #for one sample
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)] #finds the highest posterior and finds the index of it

    #gaussian function, calculates the class conditional probability
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2*var))
        denominator = np.sqrt(2* np.pi * var)
        return numerator/denominator

            

    









    # def priorProbabilities (labels):
    #     total = len(labels) #gets the total number of emails
    #     # keep = 0
    #     # discard = 0
    #     # for i in total:
    #     #     if(i==1):
    #     #         keep = keep+1
    #     #     if (i==0):
    #     #         discard = discard + 1


    #     # keep = sum(labels == 1) #calculates the total number of keep emails
    #     # discard = sum(labels == 0) #calculate the total number of discard emails 
    #     priorKeep = keep / total #calculates the prior probability for keep
    #     priorDiscard = discard / total #calculates the prior probability for discard

    #     print(total)
    #     print(keep)
    #     print(discard)
    #     print(priorKeep)
    #     print(priorDiscard)
        
    #     return priorKeep, priorDiscard

    # # def conditionalProbabilities(features, labels):
    # #     keep = sum(labels == 1) 
    # #     discard = sum(labels == 0)
        
    # #     wordIfKeep = features[labels == 1].sum() #calcualtes the num of times a word occurrs given it is a keep email
    # #     wordIfDiscard = features[labels == 0].sum() #calcualtes the num of times a word occurrs given it is a discard email
        
    # #     probWordIfKeep = wordIfKeep / keep #calculates the probability of that word given keep
    # #     probWordIfDiscard = wordIfDiscard / discard #calculates the probability of that word given discard
        
    # #     return probWordIfKeep, probWordIfDiscard



    # priorKeep, priorDiscard = priorProbabilities(body_labels)
    # # print("Prior Probability of Keep Emails is:", priorKeep)
    # # print("Prior Probability of Discard Emails is: ", priorDiscard)

    # # probWordIfKeep, probWordIfDiscard = conditionalProbabilities(body_features, body_labels)
    # # print("Conditional Probabilities of Words Given it is a Keep Email:")
    # # print(probWordIfKeep)
    # # print("\nConditional Probabilities of Words Given it is a Discard Email:")
    # # print(probWordIfDiscard)



 