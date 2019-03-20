import sys
import argparse
import pandas as pd
import numpy as np
import statistics
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Classifier.Classifier import Classifier

from sklearn.naive_bayes import MultinomialNB

import nltk

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

#import sys
#sys.stdout = open("output.txt", "a")

def main():
    start = time.time()

    ## Get Command-line Arguments #################
    parser = argparse.ArgumentParser()
    #parser.add_argument('-c', '--classifier', default='nb', help='nb | svm | dt')
    parser.add_argument('-v', '--vectorizer', default='count', help='count | tfidf | hash')
    opts = parser.parse_args()
    ###############################################

    print("Vectorizer: %s" % (opts.vectorizer))
    #print("Classifier: %s" % (opts.classifier))

    ## Build the Training Set and Testing Set #####
    training_data_dict = {}
    training_data_dict['text'] = []
    training_data_dict['length'] = []
    training_data_dict['fmeasure'] = []
    training_data_classification = []

    testing_data_dict = {}
    testing_data_dict['text'] = []
    testing_data_dict['length'] = []
    testing_data_dict['fmeasure'] = []
    testing_data_classification = []

    names = ['Classification', 'Text', 'Length', 'F-Measure']
    df = pd.read_excel('data/train_data.xlsx', header=None, names=names, usecols="A,B,C,D")

    for i in range(len(df['Text'])):
        text = df['Text'][i]
        classification = df['Classification'][i]
        text_length = df['Length'][i]
        fmeasure = df['F-Measure'][i]
        training_data_dict['text'].append(text)
        training_data_dict['length'].append(text_length)
        training_data_dict['fmeasure'].append(fmeasure)
        training_data_classification.append(classification)

    df = pd.read_excel('data/test_data.xlsx', header=None, names=names, usecols="A,B,C,D")
    for i in range(len(df['Text'])):
        text = df['Text'][i]
        classification = df['Classification'][i]
        text_length = df['Length'][i]
        fmeasure = df['F-Measure'][i]
        testing_data_dict['text'].append(text)
        testing_data_dict['length'].append(text_length)
        testing_data_dict['fmeasure'].append(fmeasure)
        testing_data_classification.append(classification)
    ###############################################

    clf = Classifier()

    features = None
    features = clf.GetFeatures(training_data_dict['text'], training_data_classification, opts.vectorizer)

    nb_clf = clf.BuildClassifier(training_data_dict, training_data_classification, opts.vectorizer, 'nb', features)
    #svm_clf = clf.RunClassifier(training_data_text, training_data_classification, opts.vectorizer, 'svm', features)
    #dt_clf = clf.RunClassifier(training_data_text, training_data_classification, opts.vectorizer, 'dt')
    #rf_clf = clf.RunClassifier(training_data_text, training_data_classification, opts.vectorizer, 'rf')
    #log_clf = clf.RunClassifier(training_data_text, training_data_classification, opts.vectorizer, 'log')

    #nb_female_acc, nb_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'nb')
    #svm_female_acc, svm_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'svm')
    #dt_female_acc, dt_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'dt')
    #rf_female_acc, rf_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'rf')
    #log_female_acc, log_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'log')

    #vectorizer = nb_clf.named_steps['vectorizer']
    #reducer = nb_clf.named_steps['reducer']
    classif = nb_clf.named_steps['clf']

    feats = nb_clf.named_steps['features']

    ## Test Model #################################
    test0 = feats.transform(testing_data_dict)
    #test1 = reducer.transform(test0)
    nb_predictions = classif.predict(test0)
    nb_predictions_2 = nb_clf.predict(testing_data_dict)

    #nb_predictions = nb_clf.predict(testing_data_text)
    #svm_predictions = svm_clf.predict(testing_data_text)
    #dt_predictions = dt_clf.predict(testing_data_text)
    #rf_predictions = rf_clf.predict(testing_data_text)
    #log_predictions = log_clf.predict(testing_data_text)

    ### Validate Model - k-fold Cross Validation ###
    print("### Cross Validation Results ###")
    print("Training Score: %f" % (nb_clf.score(training_data_dict, training_data_classification)))

    test2 = feats.transform(training_data_dict)
    #test3 = reducer.transform(test2)
    cv_scores = cross_val_score(MultinomialNB(), test2, training_data_classification, cv=10, scoring='accuracy')
    print("Cross Validation Scores:")
    print(cv_scores) 
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    print()
    ################################################

    #correct_classifications = 0
    #for (clsif, nb_pred, svm_pred, dt_pred, rf_pred, log_pred) in zip(testing_data_classification, nb_predictions, svm_predictions, dt_predictions, rf_predictions, log_predictions):
    #    male_weighted_score = 0
    #    female_weighted_score = 0

    #    if nb_pred == 1:
    #        male_weighted_score += nb_male_acc
    #    else:
    #        female_weighted_score += nb_female_acc

    #    if svm_pred == 1:
    #        male_weighted_score += svm_male_acc
    #    else:
    #        female_weighted_score += svm_female_acc

    #    if dt_pred == 1:
    #        male_weighted_score += dt_male_acc
    #    else:
    #        female_weighted_score += dt_female_acc

    #    if rf_pred == 1:
    #        male_weighted_score += rf_male_acc
    #    else:
    #        female_weighted_score += rf_female_acc

    #    if log_pred == 1:
    #        male_weighted_score += log_male_acc
    #    else:
    #        female_weighted_score += log_female_acc

    #    if male_weighted_score > female_weighted_score:
    #        # predict male
    #        if clsif == 1:
    #            correct_classifications += 1
    #    else:
    #        #predict female
    #        if clsif == 0:
    #            correct_classifications += 1

    #print("Mixed Accuracy: %0.2f" % (correct_classifications / len(testing_data_classification)))

    print("NB Accuracy: %0.2f" % (accuracy_score(testing_data_classification, nb_predictions_2)))
    print("NB Accuracy: %0.2f" % (accuracy_score(testing_data_classification, nb_predictions)))
    #print("SVM Accuracy: %0.2f" % (accuracy_score(testing_data_classification, svm_predictions)))
    #print("DT Accuracy: %0.2f" % (accuracy_score(testing_data_classification, dt_predictions)))
    #print("RF Accuracy: %0.2f" % (accuracy_score(testing_data_classification, rf_predictions)))
    #print("LOG Accuracy: %0.2f" % (accuracy_score(testing_data_classification, log_predictions)))
    #print("AVG: %0.2f" % (statistics.mean([accuracy_score(testing_data_classification, nb_predictions), 
    #                           accuracy_score(testing_data_classification, svm_predictions), 
    #                           accuracy_score(testing_data_classification, dt_predictions),
    #                           accuracy_score(testing_data_classification, rf_predictions),
    #                           accuracy_score(testing_data_classification, log_predictions)])
    #                      ))

    end = time.time()
    print("Time Run = %fs" % (end - start))

    return

if __name__ == '__main__':
    main()