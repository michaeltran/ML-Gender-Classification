#import nltk
#from nltk.util import ngrams

#def word_grams(words, min=1, max=4):
#    s = []
#    for n in range(min, max):
#        for ngram in ngrams(words, n):
#            s.append(' '.join(str(i) for i in ngram))
#    return s

#print(word_grams('one two three four'.split(' ')))

import pandas as pd

names = ['Text', 'Classification']
df = pd.read_excel('data/blog-gender-dataset.xlsx', header=None, names=names, usecols="A,B")
print(df.columns)
print(df['Classification'][0])
print(df['Text'][0])