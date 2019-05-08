import sys
import argparse
import pandas as pd
import numpy as np
import statistics
import time
import multiprocessing as mp

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from sklearn import model_selection
from scipy.sparse import coo_matrix

from Classifier.Classifier import Classifier
from Helper.DebugPrint import DebugPrint

import matplotlib.pyplot as plt
plt.style.use('ggplot')

clf = Classifier()

RANDOMIZER_SEED = 1

def main():
    start = time.time()

    ## Get Command-line Arguments #################
    parser = argparse.ArgumentParser()
    opts = parser.parse_args()
    ###############################################

    ## Build the Training Set and Testing Set #####
    training_data_dict = LoadDataSet('data/train_data.xlsx')
    testing_data_dict = LoadDataSet('data/test_data.xlsx')
    unlabeled_data_dict = LoadDataSet('data/unlabeled_data.xlsx')
    ###############################################

    ## Load POS Patterns ##########################
    pos_pattern_vocab = []
    with open('data/POSPatterns.txt') as file:
        for line in file:
            pos_pattern_vocab.append(line.strip('\n'))
    word_pattern_vocab = []
    #with open('data/WordPatterns.txt', encoding="utf8") as file:
    #    for line in file:
    #        if line.strip('\n') != '':
    #            word_pattern_vocab.append(line.strip('\n'))
    ###############################################

    predictors = {}

    print('## Initial test set accuracy...')
    svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'usl')
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM USL Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))

    print('## Adding test set back into training set...')
    # Add Testing Data Back to Training Data
    for i in range(len(testing_data_dict['index'])):
        for key in training_data_dict:
            training_data_dict[key].append(testing_data_dict[key][i])
    testing_data_dict = {}
    CrossValidationTest(training_data_dict, None, pos_pattern_vocab)
    print('## Adding some more labled data into training set...')
    # Add Unlabled Data
    for i in range(3000):
        for key in training_data_dict:
            training_data_dict[key].append(unlabeled_data_dict[key][0])
            del unlabeled_data_dict[key][0]


    ## Neural Networks and SSS Learning ###########

    #svm_clf = clf.BuildClassifierSGD(training_data_dict, pos_pattern_vocab, 'sgd')
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #print("SGD Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    #predictors['SGD'] = svm_predictions

    print('## Extracting new test set from training set...')
    # Extract Testing Data
    X_train = {}
    Y_test = {}
    kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=RANDOMIZER_SEED)
    for train_index, test_index in kf.split(training_data_dict['index']):
        for key in training_data_dict:
            X_train[key] = [training_data_dict[key][i] for i in train_index]
        for key in training_data_dict:
            Y_test[key] = [training_data_dict[key][i] for i in test_index]
        break

    training_data_dict = X_train
    testing_data_dict = Y_test

    X_train = {}
    Y_test = {}

    print('## CV Accuracy on new training/test set...')
    CrossValidationTest(training_data_dict, None, pos_pattern_vocab)

    print('## MLP Accuracy on new training/test set...')
    nb_clf = clf.BuildClassifierMLP(training_data_dict, pos_pattern_vocab, 'tf')
    nb_predictions = nb_clf.predict(testing_data_dict)
    print("MLP Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))

    print('## Keras Accuracy on new training/test set...')
    history, predictions = clf.BuildClassifierKeras(training_data_dict, testing_data_dict, pos_pattern_vocab, 'tf')

    print('## Supervised Semi-Supervised Learning...')
    training_data_dict = clf.SemiSupervisedLearning(training_data_dict, testing_data_dict, unlabeled_data_dict, pos_pattern_vocab)

    svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'usl')
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM USL Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    predictors['SVM USL'] = svm_predictions

    print('## MLP Accuracy on SSS dataset')
    nb_clf = clf.BuildClassifierMLP(training_data_dict, pos_pattern_vocab, 'tf')
    nb_predictions = nb_clf.predict(testing_data_dict)
    print("MLP Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    predictors['MLP'] = nb_predictions

    print('## Keras Accuracy on SSS dataset')
    history, predictions = clf.BuildClassifierKeras(training_data_dict, testing_data_dict, pos_pattern_vocab, 'tf')
    predictors['KERAS'] = nb_predictions

    print('## CV Accuracy on SSS dataset')
    CrossValidationTest(training_data_dict, None, pos_pattern_vocab)

    #######################

    ##svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'usl')
    ##svm_predictions = svm_clf.predict(testing_data_dict)
    ##print("SVM USL Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    ##predictors['USL'] = svm_predictions

    #CrossValidationTest(training_data_dict, None, pos_pattern_vocab)

    #training = clf.SemiSupervisedLearning(training_data_dict, testing_data_dict, unlabeled_data_dict, pos_pattern_vocab)

    #svm_clf = clf.BuildClassifierSVM(training, training['classification'], pos_pattern_vocab, word_pattern_vocab, 'usl')
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #print("SVM USL Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    #predictors['USL'] = svm_predictions

    #CrossValidationTest(training, None, pos_pattern_vocab)



    #added_blog = clf.SemiSupervisedLearning2(training_data_dict, testing_data_dict, unlabeled_data_dict, pos_pattern_vocab)

    #new_training_set = dict(added_blog)
    #for i in range(len(training_data_dict['index'])):
    #    for key in training_data_dict:
    #        new_training_set[key].append(training_data_dict[key][i])

    #CrossValidationTest(training_data_dict, added_blog, pos_pattern_vocab)

    ##svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'tf')
    ##svm_predictions = svm_clf.predict(testing_data_dict)
    ##print("SVM TF Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    ##predictors['SVM TF'] = svm_predictions

    #svm_clf = clf.BuildClassifierSVM(new_training_set, new_training_set['classification'], pos_pattern_vocab, word_pattern_vocab, 'usl')
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #print("SVM USL Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    #predictors['USL'] = svm_predictions

    #history = clf.BuildClassifierKeras(training_data_dict, testing_data_dict, pos_pattern_vocab, 'tf')

    #nb_clf = clf.BuildClassifierMLP(training_data_dict, pos_pattern_vocab, 'tf')
    #nb_predictions = nb_clf.predict(testing_data_dict)
    #print("MLP Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    #predictors['MLP'] = nb_predictions

    #svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'tf')
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #print("SVM TF Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    #predictors['SVM TF'] = svm_predictions

    #history = clf.BuildClassifierKeras(training_data_dict, testing_data_dict, pos_pattern_vocab, 'tf')

    #training = clf.SemiSupervisedLearning(training_data_dict, testing_data_dict, unlabeled_data_dict, pos_pattern_vocab)

    #CrossValidationTest(training, pos_pattern_vocab)

    #training_data_dict = training
    #history = clf.BuildClassifierKeras(training, testing_data_dict, pos_pattern_vocab, 'tf')

    #plot_history(history)

    ##svm_clf = clf.BuildClassifierSVM(training, training['classification'], pos_pattern_vocab, word_pattern_vocab, 'tf')
    ##svm_predictions = svm_clf.predict(testing_data_dict)
    ##print("SVM TF Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    ##predictors['SVM TF'] = svm_predictions


    #svm_clf = clf.BuildClassifierSVM(training, training['classification'], pos_pattern_vocab, word_pattern_vocab, 'usl')
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #print("SVM USL Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    #predictors['USL'] = svm_predictions

    #nb_clf = clf.BuildClassifierMLP(training, pos_pattern_vocab, 'tf')
    #nb_predictions = nb_clf.predict(testing_data_dict)
    #print("MLP Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    #predictors['MLP'] = nb_predictions

    ##############################################

    ## Ensemble ##################################

    ensemble_clf = clf.BuildClassifierEnsemble(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'bool-bagging')
    ensemble_predictions = ensemble_clf.predict(testing_data_dict)
    print("ENSEMBLE BAGGING Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], ensemble_predictions)))
    predictors['ENSEMBLE BAGGING'] = ensemble_predictions

    #ensemble_clf = clf.BuildClassifierEnsemble(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'discrete-bagging-r')
    #ensemble_predictions = ensemble_clf.predict(testing_data_dict)
    #predictions = []
    #for prediction in ensemble_predictions:
    #    if prediction >= 0:
    #        predictions.append(1)
    #    else:
    #        predictions.append(-1)
    #print("ENSEMBLE-R Accuracy: %0.2f" % (accuracy_score(testing_data_dict['classification'], predictions)))
    #predictors['ENSEMBLE BAGGING'] = ensemble_predictions
    ##############################################

    ## Naive Bayes ################################
    print("### Naive Bayes ###")
    nb_clf = clf.BuildClassifierNB(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'tf')
    nb_predictions = nb_clf.predict(testing_data_dict)
    print("NB TF Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    predictors['NB TF'] = nb_predictions

    nb_clf = clf.BuildClassifierNB(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'discrete')
    nb_predictions = nb_clf.predict(testing_data_dict)
    print("NB DISCRETE Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    predictors['NB DISCRETE'] = nb_predictions

    nb_clf = clf.BuildClassifierNB(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'bool')
    nb_predictions = nb_clf.predict(testing_data_dict)
    print("NB BOOL Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], nb_predictions)))
    predictors['NB BOOL'] = nb_predictions
    ##############################################

    ## SVM ########################################
    print("### SVM ###")
    svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'bool')
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM BOOL Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    predictors['SVM BOOL'] = svm_predictions

    svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'tf')
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM TF Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    predictors['SVM TF'] = svm_predictions

    #svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'svmlight-tf')
    #feats = svm_clf.named_steps['features']
    #X = feats.transform(training_data_dict)
    #ConvertToSVMLight(X, training_data_dict['classification'], 'data/TrainSVMLight.txt')
    #X = feats.transform(testing_data_dict)
    #ConvertToSVMLight(X, testing_data_dict['classification'], 'data/TestSVMLight.txt')
    #svm_predictions = svm_clf.predict(testing_data_dict)
    #print("SVM LIGHT Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    #predictors['SVM LIGHT'] = svm_predictions

    svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'discrete')
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM DISCRETE Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    predictors['SVM DISCRETE'] = svm_predictions

    svm_clf = clf.BuildClassifierSVM(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'svc')
    svm_predictions = svm_clf.predict(testing_data_dict)
    print("SVM SVC Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], svm_predictions)))
    predictors['SVM SVC'] = svm_predictions
    ###############################################

    ## SVM - Regression ##########################
    print("### SVM - Regression ###")
    svmr_clf = clf.BuildClassifierSVMR(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'linearsvr')
    svmr_predictions = svmr_clf.predict(testing_data_dict)
    predictions = []
    for prediction in svmr_predictions:
        if prediction >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    print("SVM-R LINEAR DEFAULT Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], predictions)))
    predictors['SVM-R LINEAR'] = svmr_predictions

    svmr_clf = clf.BuildClassifierSVMR(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'bool')
    svmr_predictions = svmr_clf.predict(testing_data_dict)
    predictions = []
    for prediction in svmr_predictions:
        if prediction >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    print("SVM-R BOOL Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], predictions)))
    predictors['SVM-R BOOL'] = svmr_predictions

    svmr_clf = clf.BuildClassifierSVMR(training_data_dict, training_data_dict['classification'], pos_pattern_vocab, word_pattern_vocab, 'tf')
    svmr_predictions = svmr_clf.predict(testing_data_dict)
    predictions = []
    for prediction in svmr_predictions:
        if prediction >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    print("SVM-R TF Accuracy: %0.3f" % (accuracy_score(testing_data_dict['classification'], predictions)))
    predictors['SVM-R TF'] = svmr_predictions
    ##############################################

    ## Test Model #################################
    correct_classifications, predictor_score = GetFinalPrediction(testing_data_dict['classification'], predictors)
    print("FINAL Accuracy: %0.3f" % (correct_classifications / len(testing_data_dict['classification'])))

    print(predictor_score)
    print("\n Removing worst predictor: " + predictor_score[0][0])
    del predictors[predictor_score[0][0]]

    correct_classifications, predictor_score = GetFinalPrediction(testing_data_dict['classification'], predictors)
    print("FINAL Accuracy: %0.3f" % (correct_classifications / len(testing_data_dict['classification'])))
    
    print(predictor_score)
    print("\n Removing worst predictor: " + predictor_score[0][0])
    del predictors[predictor_score[0][0]]

    correct_classifications, predictor_score = GetFinalPrediction(testing_data_dict['classification'], predictors)
    print("FINAL Accuracy: %0.3f" % (correct_classifications / len(testing_data_dict['classification'])))
    
    print(predictor_score)
    print("\n Removing worst predictor: " + predictor_score[0][0])
    del predictors[predictor_score[0][0]]

    correct_classifications, predictor_score = GetFinalPrediction(testing_data_dict['classification'], predictors)
    print("FINAL Accuracy: %0.3f" % (correct_classifications / len(testing_data_dict['classification'])))
    print(predictor_score)
    print()

    end = time.time()
    print("Time Run = %fs" % (end - start))

    return

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def GetFinalPrediction(real_classification, predictors):
    predictor_score = {}
    for predictor_type, predictor in predictors.items():
        predictor_score[predictor_type] = 0

    correct_classifications_male = 0
    correct_classifications_female = 0
    for i in range(len(real_classification)):
        actual_class = real_classification[i]
        male_vote = 0
        female_vote = 0

        for predictor_type, predictor in predictors.items():
            if predictor[i] == 1:
                male_vote += 1
            else:
                female_vote += 1

        if male_vote > female_vote and actual_class == 1:
            correct_classifications_male += 1
        elif female_vote > male_vote and actual_class == -1:
            correct_classifications_female += 1
        else:
            for predictor_type, predictor in predictors.items():
                if predictor[i] != actual_class:
                    predictor_score[predictor_type]+= 1

        if male_vote == female_vote:
            #print('Tie Breaker, Guess Male')
            if actual_class == 1:
                correct_classifications_male += 1
                for predictor_type, predictor in predictors.items():
                    if predictor[i] != actual_class:
                        predictor_score[predictor_type] += 1

    print('Correct Male: %d - Correct Female: %d' % (correct_classifications_male, correct_classifications_female))

    return correct_classifications_male + correct_classifications_female, [(k, predictor_score[k]) for k in sorted(predictor_score, key=predictor_score.get, reverse=True)]

def CrossValidationTest(X, X_addon, pos_pattern_vocab):
    kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=RANDOMIZER_SEED)

    cv_scores = []
    pool = mp.Pool(5)
    def CollectResults(result):
        cv_scores.append(result)

    for train_index, test_index in kf.split(X['index']):
        X_train = {}
        for key in X:
            X_train[key] = [X[key][i] for i in train_index]
        Y_test = {}
        for key in X:
            Y_test[key] = [X[key][i] for i in test_index]

        if X_addon is not None:
            for i in range(len(X_addon['index'])):
                for key in X_train:
                    X_train[key].append(X_addon[key][i])

        pool.apply_async(DoCVSplitTest, args=(X_train, Y_test, pos_pattern_vocab), callback=CollectResults)

    pool.close()
    pool.join()

    cv_scores = np.array(cv_scores)
    print("Cross Validation Accuracy: %0.4f (+/- %0.2f)" % (cv_scores[:,0].mean(), cv_scores[:,0].std()))
    print("Cross Validation Precision: %0.4f (+/- %0.2f)" % (cv_scores[:,1].mean(), cv_scores[:,1].std()))
    print("Cross Validation Recall: %0.4f (+/- %0.2f)" % (cv_scores[:,2].mean(), cv_scores[:,2].std()))
    print("Cross Validation F-score: %0.4f (+/- %0.2f)" % (cv_scores[:,3].mean(), cv_scores[:,3].std()))
    return cv_scores

def DoCVSplitTest(X_train, Y_test, vocab):
    classifier = clf.BuildClassifierSVM(X_train, X_train['classification'], vocab, None, 'usl')
    predictions = classifier.predict(Y_test)
    accuracy = np.array([accuracy_score(Y_test['classification'], predictions)])
    prfs = np.array(precision_recall_fscore_support(Y_test['classification'], predictions, average='weighted', warn_for=()))
    result = np.append(accuracy, prfs)
    return result

def LoadDataSet(path):
    data_dict = {}
    data_dict['index'] = []
    data_dict['classification'] = []
    data_dict['tokenized_text'] = []
    data_dict['tokenized_text_2'] = []
    data_dict['tagged_text'] = []
    data_dict['text'] = []
    data_dict['pos'] = []
    data_dict['wordcount'] = []
    data_dict['length'] = []
    data_dict['fmeasure'] = []
    data_dict['gpf'] = []
    data_dict['fa'] = []

    data_dict['le_c'] = []
    data_dict['words_misspelled'] = []
    data_dict['ts'] = []

    df = pd.read_excel(path, usecols=range(0, 60))
    for i in range(len(df['Classification'])):
        gpf = []
        fa = []
        for j in range(10):
            gpf.append(0)
        for j in range(23):
            fa.append(0)

        #text = str(df['Text'][i])
        tokenized_text = str(df['TokenizedText'][i])
        tokenized_text_2 = str(df['TokenizedText2'][i])
        tagged_text = str(df['TaggedPOS'])
        text = str(df['Text'][i])
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

        le_c = df['LE_C'][i]
        words_misspelled = df['WordsMispelled'][i] / word_count
        ts = df['TS'][i]

        data_dict['index'].append(i)
        data_dict['classification'].append(classification)
        data_dict['tokenized_text'].append(tokenized_text)
        data_dict['tokenized_text_2'].append(tokenized_text_2)
        data_dict['tagged_text'].append(tagged_text)
        data_dict['text'].append(text)
        data_dict['pos'].append(pos)
        data_dict['wordcount'].append(word_count)
        data_dict['length'].append(text_length)
        data_dict['fmeasure'].append(fmeasure)
        data_dict['gpf'].append(gpf)
        data_dict['fa'].append(fa)

        data_dict['le_c'].append(le_c)
        data_dict['words_misspelled'].append(words_misspelled)
        data_dict['ts'].append(ts)
    return data_dict

def ConvertToSVMLight(X, Y, path):
    cx = coo_matrix(X)

    with open(path, 'w') as file:
        current_i = -1
        current_line = []
        for i, j, v in zip(cx.row, cx.col, cx.data):
            if current_i != i:
                if current_line:
                    current_line_2 = sorted(current_line, key=lambda x: int(x.split(':')[0]))

                    if Y[current_i] == 1:
                        #current_line.append('+1')
                        file.write('+1 ' + ' '.join(current_line_2) + '\n')
                    else:
                        #current_line.append('-1')
                        file.write('-1 ' + ' '.join(current_line_2) + '\n')
                current_line = []
                current_i = i
            current_line.append('%d:%0.16f' % (j + 1, v))
    return

if __name__ == '__main__':
    main()