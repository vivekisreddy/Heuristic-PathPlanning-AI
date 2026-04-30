import pandas as pd
class Main:

    bodies_data = pd.read_csv("dbworld_bodies_stemmed.csv", header = None) #puts the bodies data in to a dataframe to access easier
    subjects_data = pd.read_csv("dbworld_bodies_stemmed.csv", header = None) #puts the bodies data into a dataframe to access easier 
    

    bodies_data = bodies_data.drop(columns=0) #removes the id column
    subjects_data = subjects_data.drop(columns=0)

    body_labels = bodies_data.iloc[:, -1]  # Separates labels (CLASS) - if the email should be discarded or not
    subject_labels = subjects_data.iloc[:, -1]

    body_features = bodies_data.iloc[:, :-1]  # Separates features (words) for both datasets 
    subject_features = subjects_data.iloc[:, :-1]  

    
    
    #counts the num of email that are spam
    subject_discard = 0
    body_discard = 0
    for i in range(len(subject_labels)-1): 
        # print(subject_labels[i+1])
        if subject_labels[i+1]=="0": 
            subject_discard = subject_discard + 1

    #COUNTS THEE NUM SPAM IN THE BODY LABELS - SAME AS THE NUM OF SPAM IN LABEL
    # for i in range(len(body_labels)-1):
    #     if body_labels[i+1]=="0":
    #         body_discard = body_discard + 1
    # print(body_discard)

    discardCount = subject_discard

    #calculates the num of times the word appeare in emails that won't be discarded
    keepCount = 0
    for i in range(len(subject_labels)-1): 
        # print(subject_labels[i+1])
        if subject_labels[i+1]=="1": 
            keepCount = keepCount + 1


    #MIGHT NOT EVEN NEED THIS TBH BUT KEEPING FOR NOW
    discardPriot = discardCount/len(subject_labels)
    keepPrior = keepCount/len(subject_labels)

    #finds the frequency of words in both spam and not spam in the subject
    count1 = 0
    count2 = 0
    subjectfeatureOccurranceCount = []
    for i in range(subject_features.shape[1]):
        for j in range(len(subject_features)):
            if subject_labels[j]=="0": #finds the frequency of the word in spam
                if subject_features.iloc[j,i] == "1":
                    count1 = count1 + 1
            elif subject_labels[j]=="1": #finds the frequency of the word not in spam
                if subject_features.iloc[j,i] == "1": #finding the frequency of the word in regular emails
                    count2 = count2 + 1
        subjectfeatureOccurranceCount.append((subject_features.iloc[0,i],(count1/len(subject_features)),(count2/len(subject_features)))) #name, spamfreq, notspamfreq
        count1 = 0
        count2 = 0
    # print(subjectfeatureOccurranceCount)


    #finds the frequency of words in both spam and not spam in the body
    count1 = 0
    count2 = 0
    bodyfeatureOccurranceCount = []
    for i in range(body_features.shape[1]):
        for j in range(len(body_features)):
            if body_labels[j]=="0": #finds the frequency of the word in spam
                if body_features.iloc[j,i] == "1":
                    count1 = count1 + 1
            elif body_labels[j]=="1": #finds the frequency of the word not in spam
                if body_features.iloc[j,i] == "1": #finding the frequency of the word in regular emails
                    count2 = count2 + 1
        bodyfeatureOccurranceCount.append((body_features.iloc[0,i],(count1/len(subject_features)),(count2/len(subject_features))))
        count1 = 0
        count2 = 0
    # print(bodyfeatureOccurranceCount)

    #APPLY LAPLAS FORMULA FOR WORDS THAT HAVE NOT BEEN SEEN







    
