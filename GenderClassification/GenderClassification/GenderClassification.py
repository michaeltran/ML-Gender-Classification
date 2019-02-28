import argparse
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    ## Get Command-line Arguments #################
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classifier', default='nb', help='nb | svm | dt')
    parser.add_argument('-v', '--vectorizer')
    opts = parser.parse_args()
    ###############################################

    ## Build the Training Set and Testing Set #####
    names = ['Text', 'Classification']
    df = pd.read_excel('data/blog-gender-dataset.xlsx', header=None, names=names, usecols="A,B")

    training_data_text = []
    training_data_classification = []
    testing_data_text = []
    testing_data_classification = []

    # Prepare and Sanitize data
    for i in range(len(df['Text'])):
        text = df['Text'][i]
        classification = df['Classification'][i]
        if text == text and classification == classification:
            training_data_text.append(text)
            classification = classification.strip().upper()
            if classification == 'M':
                training_data_classification.append(1)
            elif classification == 'F':
                training_data_classification.append(0)
            else:
                print('Classification Error: %s is not defined.' % (classification))
                return

    # Split out a Testing dataset
    len_testing_data = int(len(training_data_text)/10)
    for i in range(len_testing_data):
        random_index = random.randint(0, len(training_data_text) - 1)
        testing_data_text.append(training_data_text[random_index])
        testing_data_classification.append(training_data_classification[random_index])
        del training_data_text[random_index]
        del training_data_classification[random_index]
    ###############################################

    ## Build and Train Model ######################
    vectorizer = CountVectorizer()
    if opts.classifier == 'nb':
        classifier = MultinomialNB()
    elif opts.classifier == 'svm':
        classifier = LinearSVC()
    elif opts.classifier == 'dt':
        classifier = DecisionTreeClassifier()

    text_clf = Pipeline([
            ('vect', vectorizer),
            ('clf', classifier)
        ])
    text_clf.fit(training_data_text, training_data_classification)
    ###############################################

    ## Validate Model - k-fold Cross Validation ###
    print("Cross Validation Results")
    print("Training Score: %f" % (text_clf.score(training_data_text, training_data_classification)))

    cv_scores = cross_val_score(text_clf, training_data_text, training_data_classification, cv=10, scoring='accuracy')
    print("Cross Validation Scores:")
    print(cv_scores) 
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    print()
    ###############################################

    ## Test Model #################################
    print("Test Results")
    print("Testing Score: %f" % (text_clf.score(testing_data_text, testing_data_classification)))
    
    predictions = text_clf.predict(testing_data_text)
    #print("Classification report:")
    #print(classification_report(testing_data_classification, predictions))

    #print("Confusion Matrix:")
    #print(confusion_matrix(testing_data_classification, predictions))

    print("Testing Accuracy: %0.2f" % (accuracy_score(testing_data_classification, predictions)))

    #test_predicted_proba = text_clf.predict_proba(testing_data_text)
    #TP = 0
    #TN = 0
    #FP = 0
    #FN = 0

    #for (a, b) in zip(testing_data_classification, predictions):
    #    if a == b:
    #        if a == 1:
    #            TP += 1
    #        else:
    #            TN += 1
    #    else:
    #        if a == 1:
    #            FN += 1
    #        else:
    #            FP += 1
    #    print("%d %d" % (a, b))
    ###############################################

    return

if __name__ == '__main__':
    main()