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

from sklearn import model_selection

from Classifier.Classifier import Classifier
from Helper.DebugPrint import DebugPrint

clf = Classifier()

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
    word_pattern_vocab = []
    with open('data/WordPatterns.txt') as file:
        for line in file:
            if line.strip('\n') != '':
                word_pattern_vocab.append(line.strip('\n'))
    ###############################################

    nb_classifiers = []
    nb_predictors = []

    svm_classifiers = []
    svm_predictors = []

    svmr_classifiers = []
    svmr_predictors = []

    #svmr_clf = clf.BuildClassifierSVMR(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'bool')
    #svmr_predictions = svmr_clf.predict(testing_data_dict)
    #predictions = []
    #for prediction in svmr_predictions:
    #    if prediction >= 0.5:
    #        predictions.append(1)
    #    else:
    #        predictions.append(0)
    #print("SVMR DEFAULT Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], predictions)))
    #svmr_classifiers.append(svmr_clf)
    #svmr_predictors.append(predictions)

    #svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'svc')
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #print("SVM SVC Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    #svm_classifiers.append(svm_clf)
    #svm_predictors.append(svm_predictions)

    #svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'default')
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #print("SVM DEFAULT Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    #svm_classifiers.append(svm_clf)
    #svm_predictors.append(svm_predictions)

    #nb_clf = clf.BuildClassifierNB(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'default')
    #nb_predictions = nb_clf.predict(testing_data_dict)
    #print("NB DEFAULT Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    #nb_classifiers.append(nb_clf)
    #nb_predictors.append(nb_predictions)

    ## Naive Bayes ################################
    print("### Naive Bayes ###")
    nb_clf = clf.BuildClassifierNB(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'tf')
    nb_predictions = nb_clf.predict(testing_data_dict)
    print("NB TF Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    nb_classifiers.append(nb_clf)
    nb_predictors.append(nb_predictions)

    nb_clf = clf.BuildClassifierNB(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'discrete')
    nb_predictions = nb_clf.predict(testing_data_dict)
    print("NB DISCRETE Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    nb_classifiers.append(nb_clf)
    nb_predictors.append(nb_predictions)

    nb_clf = clf.BuildClassifierNB(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'bool')
    nb_predictions = nb_clf.predict(testing_data_dict)
    print("NB BOOL Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    nb_classifiers.append(nb_clf)
    nb_predictors.append(nb_predictions)

    #CrossValidationTest(training_data_dict, pos_pattern_vocab)

    ## Cross Validation
    #feats = nb_clf.named_steps['features']
    #print("Training Score: %f" % (nb_clf.score(training_data_dict, training_data_dict['classification'])))
    #X = feats.transform(training_data_dict)
    #cv_scores = cross_val_score(MultinomialNB(), X, training_data_dict['classification'], cv=10, scoring='accuracy')
    #print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    #print()
    ##############################################

    ## SVM ########################################
    print("### SVM ###")
    svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'tf')
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM TF Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    svm_classifiers.append(svm_clf)
    svm_predictors.append(svm_predictions)

    svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'discrete')
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM DISCRETE Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    svm_classifiers.append(svm_clf)
    svm_predictors.append(svm_predictions)

    svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'svc')
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM SVC Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    svm_classifiers.append(svm_clf)
    svm_predictors.append(svm_predictions)

    ## Cross Validation
    #feats = svm_clf.named_steps['features']
    #print("Training Score: %f" % (svm_clf.score(training_data_dict, training_data_dict['classification'])))
    #X = feats.transform(training_data_dict)
    #cv_scores = cross_val_score(LinearSVC(max_iter=10000), X, training_data_dict['classification'], cv=10, scoring='accuracy')
    #print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    #print()
    ###############################################

    ## SVM - Regression ##########################
    print("### SVM - Regression ###")
    svmr_clf = clf.BuildClassifierSVMR(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'linearsvr')
    svmr_predictions = svmr_clf.predict(testing_data_dict)
    predictions = []
    for prediction in svmr_predictions:
        if prediction >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    print("SVM-R LINEAR DEFAULT Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], predictions)))
    svmr_classifiers.append(svmr_clf)
    svmr_predictors.append(predictions)

    svmr_clf = clf.BuildClassifierSVMR(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'bool')
    svmr_predictions = svmr_clf.predict(testing_data_dict)
    predictions = []
    for prediction in svmr_predictions:
        if prediction >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    print("SVM-R BOOL Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], predictions)))
    svmr_classifiers.append(svmr_clf)
    svmr_predictors.append(predictions)

    svmr_clf = clf.BuildClassifierSVMR(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'tf')
    svmr_predictions = svmr_clf.predict(testing_data_dict)
    predictions = []
    for prediction in svmr_predictions:
        if prediction >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    print("SVM-R TF Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], predictions)))
    svmr_classifiers.append(svmr_clf)
    svmr_predictors.append(predictions)

    ## Cross Validation
    #feats = svm_clf.named_steps['features']
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

    correct_classifications = 0
    for i in range(len(testing_data_dict['classification'])):
        actual_class = testing_data_dict['classification'][i]
        male_vote = 0
        female_vote = 0
        for nb_predict in nb_predictors:
            if nb_predict[i] == 1:
                male_vote += 1
            else:
                female_vote += 1
        for svm_predict in svm_predictors:
            if svm_predict[i] == 1:
                male_vote += 1
            else:
                female_vote += 1
        for svmr_predict in svmr_predictors:
            if svmr_predict[i] == 1:
                male_vote += 1
            else:
                female_vote += 1
        if male_vote > female_vote and actual_class == 1:
            correct_classifications += 1
        elif female_vote > male_vote and actual_class == 0:
            correct_classifications += 1
        elif male_vote == female_vote:
            print('problemo')
            if actual_class == 1:
                correct_classifications += 1

    print("FINAL Accuracy: %0.2f" % (correct_classifications / len(testing_data_dict['classification'])))

    print()

    end = time.time()
    print("Time Run = %fs" % (end - start))

    return

def CrossValidationTest(X, pos_pattern_vocab):
    kf = model_selection.KFold(n_splits=10)

    for train_index, test_index in kf.split(X['index']):
        X_train = {}
        for key in X:
            X_train[key] = [X[key][i] for i in train_index]
        Y_test = {}
        for key in X:
            Y_test[key] = [X[key][i] for i in test_index]

        nb_clf = clf.BuildClassifierNB(X_train, X_train['classification'], None, pos_pattern_vocab)
        nb_predictions = nb_clf.predict(Y_test)
        print("NB Accuracy: %0.2f" % (accuracy_score(Y_test['classification'], nb_predictions)))

def LoadDataSet(path):
    data_dict = {}
    data_dict['index'] = []
    data_dict['classification'] = []
    data_dict['text'] = []
    data_dict['untagged_text'] = []
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

        untagged_text = df['Text'][i]
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

        data_dict['index'].append(i)
        data_dict['classification'].append(classification)
        data_dict['untagged_text'].append(untagged_text)
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