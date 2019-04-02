from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

keys = ['little inspiration', 'time', 'occasion', 'creativity', 'innovation']

text = ('Everyone needs little inspiration from time to time')

cv1 = CountVectorizer(ngram_range=(1, 2) )
data = cv1.fit_transform([text]).toarray()
#vec1 = np.array(data)
#print(vec1)

#print(cv1.get_feature_names())

#text2 = ('needs everyone')
#data = cv1.transform()

for (count, feature) in zip(data[0], cv1.get_feature_names()):
    print(count, feature)