import string
import nltk
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn

class NLTKPreprocessor():
    def __init__(self, lower=True):
        self.lower = lower
        self.stopwords = set(sw.words('english'))
        self.punct = set(string.punctuation)
        self.lemmatizer = nltk.WordNetLemmatizer()

    def TokenizeText(self, text):
        tokens = []

        for token, tag in nltk.pos_tag(nltk.wordpunct_tokenize(text)):
            #token = token.lower()
            #token = token.strip()
            #token = token.strip('_')
            #token = token.strip('*')

            if token in self.stopwords:
                continue

            #if all(char in self.punct for char in token):
            #    continue

            lemma = self.lemmatize(token, tag)

            tokens.append(lemma)

        return tokens

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'B': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)