import sys
import argparse
import pandas as pd
import numpy as np
import statistics
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from Classifier.Classifier import Classifier

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    training_data_dict, training_data_classification = LoadDataSet('data/train_data.xlsx')
    testing_data_dict, testing_data_classification = LoadDataSet('data/test_data.xlsx')
    ###############################################

    ## Load POS Patterns ##########################
    with open('data/POSPatterns.txt') as file:
        pos_patterns = file.readline()

    vocab = {}
    i = 0
    for f in pos_patterns.split('||'):
        vocab[f] = i
        i += 1
    ###############################################

    clf = Classifier()

    features = None
    features = clf.GetFeatures(training_data_dict, training_data_classification, opts.vectorizer, vocab)

    nb_clf = clf.BuildClassifier(training_data_dict, training_data_classification, opts.vectorizer, 'nb', features, vocab)
    #svm_clf = clf.BuildClassifier(training_data_dict, training_data_classification, opts.vectorizer, 'svm', features)
    #dt_clf = clf.BuildClassifier(testing_data_dict, training_data_classification, opts.vectorizer, 'dt')
    #rf_clf = clf.BuildClassifier(testing_data_dict, training_data_classification, opts.vectorizer, 'rf')
    #log_clf = clf.BuildClassifier(testing_data_dict, training_data_classification, opts.vectorizer, 'log')

    #nb_female_acc, nb_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'nb')
    #svm_female_acc, svm_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'svm')
    #dt_female_acc, dt_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'dt')
    #rf_female_acc, rf_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'rf')
    #log_female_acc, log_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'log')

    #vectorizer = nb_clf.named_steps['vectorizer']
    classif = nb_clf.named_steps['clf']
    feats = nb_clf.named_steps['features']
    #reducer = nb_clf.named_steps['reducer']

    ## Test Model #################################
    #test0 = feats.transform(testing_data_dict)
    #print('Testing Vect Examples')
    #print(test0[0])
    #print(test0[1])
    #test0 = reducer.transform(test0)
    #nb_predictions = nb_clf.predict(testing_data_dict)

    #nb_predictions = nb_clf.predict(testing_data_dict)
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #dt_predictions = dt_clf.predict(testing_data_dict)
    #rf_predictions = rf_clf.predict(testing_data_dict)
    #log_predictions = log_clf.predict(testing_data_dict)

    ## Validate Model - k-fold Cross Validation ###
    print("### Cross Validation Results ###")
    print("Training Score: %f" % (nb_clf.score(training_data_dict, training_data_classification)))
    test2 = feats.transform(training_data_dict)
    cv_scores = cross_val_score(MultinomialNB(), test2, training_data_classification, cv=10, scoring='accuracy')
    print("Cross Validation Scores:")
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    print()
    ##############################################

    ## Validate Model - k-fold Cross Validation ###
    print("### Cross Validation Results ###")
    test2 = feats.transform(training_data_dict)
    cv_scores = cross_val_score(SVC(kernel='linear'), test2, training_data_classification, cv=10, scoring='accuracy')
    print("Cross Validation Scores:")
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    print()
    ##############################################

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

    nb_predictions = nb_clf.predict(testing_data_dict)

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

    print()

    svm_clf = clf.BuildClassifier(training_data_dict, training_data_classification, opts.vectorizer, 'svm', features, vocab)
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM Accuracy: %0.2f" % (accuracy_score(testing_data_classification, svm_predictions)))

    end = time.time()
    print("Time Run = %fs" % (end - start))

    return

def LoadDataSet(path):
    data_dict = {}
    data_dict['text'] = []
    data_dict['length'] = []
    data_dict['fmeasure'] = []
    data_dict['gpf'] = []
    data_dict['fa'] = []
    data_classification = []

    df = pd.read_excel(path, usecols="A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ,AK,AL")
    
    for i in range(len(df['Text'])):
        gpf = []
        fa = []
        for j in range(10):
            gpf.append(0)
        for j in range(23):
            fa.append(0)

        #text = df['Text'][i]
        text = df['POS'][i]
        classification = df['Classification'][i]
        text_length = df['Length'][i]
        fmeasure = df['F-Measure'][i]
        gpf[0] = df['GPF1'][i]
        gpf[1] = df['GPF2'][i]
        gpf[2] = df['GPF3'][i]
        gpf[3] = df['GPF4'][i]
        gpf[4] = df['GPF5'][i]
        gpf[5] = df['GPF6'][i]
        gpf[6] = df['GPF7'][i]
        gpf[7] = df['GPF8'][i]
        gpf[8] = df['GPF9'][i]
        gpf[9] = df['GPF10'][i]
        fa[0] = df['FA1'][i]
        fa[1] = df['FA2'][i]
        fa[2] = df['FA3'][i]
        fa[3] = df['FA4'][i]
        fa[4] = df['FA5'][i]
        fa[5] = df['FA6'][i]
        fa[6] = df['FA7'][i]
        fa[7] = df['FA8'][i]
        fa[8] = df['FA9'][i]
        fa[9] = df['FA10'][i]
        fa[10] = df['FA11'][i]
        fa[11] = df['FA12'][i]
        fa[12] = df['FA13'][i]
        fa[13] = df['FA14'][i]
        fa[14] = df['FA15'][i]
        fa[15] = df['FA16'][i]
        fa[16] = df['FA17'][i]
        fa[17] = df['FA18'][i]
        fa[18] = df['FA19'][i]
        fa[19] = df['FA20'][i]
        fa[20] = df['FA21'][i]
        fa[21] = df['FA22'][i]
        fa[22] = df['FA23'][i]
        data_dict['text'].append(text)
        data_dict['length'].append(text_length)
        data_dict['fmeasure'].append(fmeasure)
        data_dict['gpf'].append(gpf)
        data_dict['fa'].append(fa)
        data_classification.append(classification)
    return data_dict, data_classification

if __name__ == '__main__':
    main()