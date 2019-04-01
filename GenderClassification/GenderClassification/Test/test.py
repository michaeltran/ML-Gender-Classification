# N-GRAM EXAMPLE
#import nltk
#from nltk.util import ngrams
#def word_grams(words, min=1, max=4):
#    s = []
#    for n in range(min, max):
#        for ngram in ngrams(words, n):
#            s.append(' '.join(str(i) for i in ngram))
#    return s
#print(word_grams('one two three four'.split(' ')))




#vectorizer = CountVectorizer()
#vectorizer.fit(data_text)

##print(vectorizer.vocabulary_)

#vector = vectorizer.transform(data_text)
##print(vector.shape)
##print(type(vector))
##print(vector.toarray())

#clf = MultinomialNB().fit(vector, data_classification)

#docs_new = [data_text[4]]
#new_counts = vectorizer.transform(docs_new)
#predicted = clf.predict(new_counts)

#for doc, category in zip(docs_new, predicted):
#    print('%r => %s' % (doc, category))




    #testing_data_text = []
    #testing_data_classification = []

    #len_testing_data = int(len(training_data_text)/10)

    #for i in range(len_testing_data):
    #    random_index = random.randint(0, len(training_data_text))
    #    testing_data_text.append(training_data_text[random_index])
    #    testing_data_classification.append(training_data_classification[random_index])
    #    del training_data_text[random_index]
    #    del training_data_classification[random_index]




    ################################################

    ### Validate Model - k-fold Cross Validation ###
    #print("### Cross Validation Results ###")
    #print("Training Score: %f" % (text_clf.score(training_data_text, training_data_classification)))

    #cv_scores = cross_val_score(text_clf, training_data_text, training_data_classification, cv=10, scoring='accuracy')
    #print("Cross Validation Scores:")
    #print(cv_scores) 
    #print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std()))
    #print()
    ################################################

    ### Test Model #################################
    #print("### Testing Results ###")
    #print("Testing Score: %f" % (text_clf.score(testing_data_text, testing_data_classification)))
    
    #predictions = text_clf.predict(testing_data_text)
    #print("Testing Accuracy: %0.2f" % (accuracy_score(testing_data_classification, predictions)))

    #print("Classification Report:")
    #print(classification_report(testing_data_classification, predictions))

    #print("Confusion Matrix:")
    #print(confusion_matrix(testing_data_classification, predictions))
    ################################################



## CHI STATISTIC MANUALLY
        #test1 = np.extract(Y, X[:,1])
        #test = np.where(test1 > 0)
        #W = float(test[0].size)

        #test = np.where(X[:,1] > 0)
        #x = float(test[0].size - W)

        #test = np.where(test1 == 0)
        #y = float(test[0].size)

        #test = np.where(X[:,1] == 0)
        #Z = float(test[0].size - y)

        #N = W + x + y + Z

        #chiiii = (N * np.power((W * Z - y * x), 2)) / ((W + y) * (x + Z) * (W + x) * (y + Z))








    # ENSEMBLE LEARNING

    #nb_female_acc, nb_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'nb')
    #svm_female_acc, svm_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'svm')
    #dt_female_acc, dt_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'dt')
    #rf_female_acc, rf_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'rf')
    #log_female_acc, log_male_acc = clf.CrossValidationTest(training_data_text, training_data_classification, opts.vectorizer, 'log')

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


    #def CrossValidationTest(self, X, Y, vectorizer_type, classifier_type):
    #    kf = model_selection.KFold(n_splits=10)

    #    female_true = 0
    #    female_false = 0
    #    male_true = 0
    #    male_false = 0

    #    for train_index, test_index in kf.split(X):
    #        X = np.array(X, dtype=object)
    #        Y = np.array(Y, dtype=object)
    #        X_train, X_test = list(X[train_index]), list(X[test_index])
    #        Y_train, Y_test = list(Y[train_index]), list(Y[test_index])

    #        clf = self.BuildClassifier(X_train, Y_train, vectorizer_type, classifier_type)

    #        conf_matrix = confusion_matrix(Y_test, clf.predict(X_test))
    #        female_true += conf_matrix[0][0]
    #        female_false += conf_matrix[1][0]
    #        male_true += conf_matrix[1][1]
    #        male_false += conf_matrix[0][1]

    #    female_accuracy = female_true / (female_false)
    #    male_accuracy = male_true / (male_false)
    #    #print("female accuracy", female_accuracy)
    #    #print("male accuracy", male_accuracy)

    #    return female_accuracy, male_accuracy