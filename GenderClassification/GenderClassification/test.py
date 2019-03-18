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


