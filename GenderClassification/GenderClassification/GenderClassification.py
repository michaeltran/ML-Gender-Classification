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

UseTF = False

def main():
    start = time.time()

    ## Get Command-line Arguments #################
    parser = argparse.ArgumentParser()
    opts = parser.parse_args()
    ###############################################

    ## Build the Training Set and Testing Set #####
    training_data_dict = LoadDataSet('data/train_data.xlsx')
    testing_data_dict = LoadDataSet('data/test_data.xlsx')
    ###############################################

    ## Load POS Patterns ##########################
    pos_pattern_vocab = []
    with open('data/POSPatterns.txt') as file:
        for line in file:
            pos_pattern_vocab.append(line.strip('\n'))
    ###############################################

    clf = Classifier()

    features = None
    #features = clf.GetFeatures(training_data_dict, training_data_dict['classification'], pos_pattern_vocab)

    ## Naive Bayes ################################
    print("### Naive Bayes ###")
    nb_clf = clf.BuildClassifierNB(training_data_dict, training_data_dict['classification'], features, pos_pattern_vocab)
    nb_predictions = nb_clf.predict(testing_data_dict)
    print("NB Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))

    feats = nb_clf.named_steps['features']

    # Cross Validation
    print("Training Score: %f" % (nb_clf.score(training_data_dict, training_data_dict['classification'])))
    X = feats.transform(training_data_dict)
    cv_scores = cross_val_score(MultinomialNB(), X, training_data_dict['classification'], cv=10, scoring='accuracy')
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    print()
    ##############################################


    ## SVM - Regression ##########################
    print("### SVM - Regression ###")
    svmr_clf = clf.BuildClassifierSVMR(training_data_dict, training_data_dict['classification'], features, pos_pattern_vocab, UseTF)
    svmr_predictions = svmr_clf.predict(testing_data_dict)

    predictions = []
    for prediction in svmr_predictions:
        if prediction >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    print("SVMR Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], predictions)))

    #feats = svm_clf.named_steps['features']

    ## Cross Validation
    #print("Training Score: %f" % (svm_clf.score(training_data_dict, training_data_dict['classification'])))
    #X = feats.transform(training_data_dict)
    #cv_scores = cross_val_score(LinearSVC(max_iter=100000), X, training_data_dict['classification'], cv=10, scoring='accuracy')
    #print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    #print()
    ##############################################

    ## Test Model #################################
    #nb_predictions = nb_clf.predict(testing_data_dict)
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #dt_predictions = dt_clf.predict(testing_data_dict)
    #rf_predictions = rf_clf.predict(testing_data_dict)
    #log_predictions = log_clf.predict(testing_data_dict)

    #print("NB Accuracy: %0.2f" % (accuracy_score(testing_data_classification, nb_predictions)))
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

    end = time.time()
    print("Time Run = %fs" % (end - start))

    return

def LoadDataSet(path):
    data_dict = {}
    data_dict['classification'] = []
    data_dict['text'] = []
    data_dict['pos'] = []
    data_dict['wordcount'] = []
    data_dict['length'] = []
    data_dict['fmeasure'] = []
    data_dict['gpf'] = []
    data_dict['fa'] = []

    df = pd.read_excel(path, usecols=range(0, 40))
    for i in range(len(df['Classification'])):
        gpf = []
        fa = []
        for j in range(10):
            gpf.append(0)
        for j in range(23):
            fa.append(0)

        #text = df['Text'][i]
        text = df['TaggedPOS'][i]
        pos = df['POS'][i]
        classification = df['Classification'][i]
        word_count = df['WordCount'][i]
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

        #gpf = [x / word_count for x in gpf]
        #fa = [x / word_count for x in fa]

        data_dict['classification'].append(classification)
        data_dict['text'].append(text)
        data_dict['pos'].append(pos)
        data_dict['wordcount'].append(word_count)
        data_dict['length'].append(text_length)
        data_dict['fmeasure'].append(fmeasure)
        data_dict['gpf'].append(gpf)
        data_dict['fa'].append(fa)
    return data_dict

if __name__ == '__main__':
    main()