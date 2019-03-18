import argparse
import pandas as pd
import numpy as np
import statistics
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Classifier.Classifier import Classifier

import nltk

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

def posNgrams(s,n):
    '''Calculate POS n-grams and return a dictionary'''
    text = nltk.word_tokenize(s)
    text_tags = nltk.pos_tag(text)
    taglist = []
    output = {}
    for item in text_tags: 
        taglist.append(item[1])
    for i in range(len(taglist)-n+1):
        g = ' '.join(taglist[i:i+n])
        output.setdefault(g,0)
        output[g] += 1
    return output

def main():

    start = time.time()

    test = posNgrams("it is sunny. it is sunny. out today", 3)

    ## Get Command-line Arguments #################
    parser = argparse.ArgumentParser()
    #parser.add_argument('-c', '--classifier', default='nb', help='nb | svm | dt')
    parser.add_argument('-v', '--vectorizer', default='count', help='count | tfidf | hash')
    opts = parser.parse_args()
    ###############################################

    print("Vectorizer: %s" % (opts.vectorizer))
    #print("Classifier: %s" % (opts.classifier))

    ## Build the Training Set and Testing Set #####
    training_data_text = []
    training_data_classification = []
    testing_data_text = []
    testing_data_classification = []

    names = ['Classification', 'Text']
    df = pd.read_excel('data/train_data.xlsx', header=None, names=names, usecols="A,B")

    for i in range(len(df['Text'])):
        text = df['Text'][i]
        classification = df['Classification'][i]
        training_data_text.append(text)
        training_data_classification.append(classification)

    df = pd.read_excel('data/test_data.xlsx', header=None, names=names, usecols="A,B")
    for i in range(len(df['Text'])):
        text = df['Text'][i]
        classification = df['Classification'][i]
        testing_data_text.append(text)
        testing_data_classification.append(classification)
    ###############################################

    clf = Classifier()

    features = None
    features = clf.GetFeatures(training_data_text, training_data_classification, opts.vectorizer)

    nb_clf = clf.RunClassifier(training_data_text, training_data_classification, opts.vectorizer, 'nb', features)
    svm_clf = clf.RunClassifier(training_data_text, training_data_classification, opts.vectorizer, 'svm', features)
    #dt_clf = clf.RunClassifier(training_data_text, training_data_classification, opts.vectorizer, 'dt')
    #rf_clf = clf.RunClassifier(training_data_text, training_data_classification, opts.vectorizer, 'rf')
    #log_clf = clf.RunClassifier(training_data_text, training_data_classification, opts.vectorizer, 'log')

    #nb_female_acc, nb_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'nb')
    #svm_female_acc, svm_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'svm')
    #dt_female_acc, dt_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'dt')
    #rf_female_acc, rf_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'rf')
    #log_female_acc, log_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'log')

    ## Test Model #################################
    nb_predictions = nb_clf.predict(testing_data_text)
    svm_predictions = svm_clf.predict(testing_data_text)
    #dt_predictions = dt_clf.predict(testing_data_text)
    #rf_predictions = rf_clf.predict(testing_data_text)
    #log_predictions = log_clf.predict(testing_data_text)

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

    print("NB Accuracy: %0.2f" % (accuracy_score(testing_data_classification, nb_predictions)))
    print("SVM Accuracy: %0.2f" % (accuracy_score(testing_data_classification, svm_predictions)))
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