import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

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

    ## Test Model #################################
    #print(text_clf.score(training_data_text, training_data_classification))

    cv_scores = cross_val_score(text_clf, training_data_text, training_data_classification, cv=10, scoring='accuracy')
    print("Cross Validation Scores:")
    print(cv_scores) 
    print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    ###############################################

    return

if __name__ == '__main__':
    main()