class ColumnExtractor(object):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        sliced = X[:, self.cols]
        return sliced

    def fit(self, X, y=None):
        return self