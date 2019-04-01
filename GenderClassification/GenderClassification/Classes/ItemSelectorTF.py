
from sklearn.base import BaseEstimator, TransformerMixin

class ItemSelectorTF(BaseEstimator, TransformerMixin):
    def __init__(self, key, keycount):
        self.key = key
        self.keycount = keycount

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        ret = []

        for i in range(len(data_dict[self.key])):
            row = data_dict[self.key][i]
            word_count = data_dict[self.keycount][i]
            ret.append([x / word_count for x in row])

        return ret