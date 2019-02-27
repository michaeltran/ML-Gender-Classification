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