import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import mutual_info_classif

from Classes.ColumnExtractor import ColumnExtractor
from Classes.EFS import EFS
from Classes.ItemSelector import ItemSelector

efs_obj = EFS()

class Classifier(object):
    def GetVectorizer(self, vectorizer_type):
        vectorizer = None
        if vectorizer_type == 'count':
            vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False)
            #vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split('||'), lowercase=False)
            #vectorizer = CountVectorizer(analyzer='word', binary=True, ngram_range=(1, 1))
        elif vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
        elif vectorizer_type == 'hash':
            vectorizer = HashingVectorizer(non_negative=True) # This seems like it isn't good
        else:
            print('Unknown Vectorizer: %s' % (vectorizer_type))
        return vectorizer

    def GetClassifier(self, classifier_type):
        classifier = None
        if classifier_type == 'nb':
            classifier = MultinomialNB()
        elif classifier_type == 'svm':
            classifier = SVC(kernel='linear')
            #classifier = LinearSVC()
        elif classifier_type == 'dt':
            classifier = DecisionTreeClassifier()
        elif classifier_type == 'rf':
            classifier = RandomForestClassifier(n_estimators=10)
        elif classifier_type == 'log':
            classifier = LogisticRegression(C=.088)
        else:
            print('Unknown Classifier: %s' % (classifier_type))
        return classifier

    def GetFeatures(self, training_data_dict, training_data_classification, vectorizer_type, vocab):
        #vectorizer = self.GetVectorizer(vectorizer_type)
        vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False)

        vectorizer_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('vectorizer', vectorizer),
                ])),
            ]))
        ])

        X = vectorizer_pipeline.fit_transform(training_data_dict, training_data_classification)
        candidate_feature_indexes = efs_obj.EFS(X.toarray(), training_data_classification)

        return candidate_feature_indexes

    def BuildClassifier(self, training_data_dict, training_data_classification, vectorizer_type, classifier_type, features, vocab):
        vectorizer = None
        reducer = None
        classifier = None

        ## Build and Train Model ######################
        #vectorizer = self.GetVectorizer(vectorizer_type)
        reducer = ColumnExtractor(cols=features)
        classifier = self.GetClassifier(classifier_type)
        vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False)
        #print(vectorizer.get_feature_names())



        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('vectorizer', vectorizer),
                    ('reducer', reducer),
                ])),
                ('length', Pipeline([
                    ('selector', ItemSelector(key='length')),
                    ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ('discretize', KBinsDiscretizer(n_bins = 100, encode='ordinal', strategy='uniform')),
                    ('toint', FunctionTransformer(self.GetIntArray, validate = False)),
                ])),
                ('fmeasure', Pipeline([
                    ('selector', ItemSelector(key='fmeasure')),
                    ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ('toint', FunctionTransformer(self.GetIntArray, validate = False)),
                ])),
                ('gpf', Pipeline([
                    ('selector', ItemSelector(key='gpf')),
                    ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                ])),
                ('fa', Pipeline([
                    ('selector', ItemSelector(key='fa')),
                    ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                ])),
            ])),
            ('clf', classifier),
        ])

        text_clf.fit(training_data_dict, training_data_classification)
        ###############################################

        #feats = text_clf.named_steps['features']
        #test = feats.transform(training_data_dict)
        #print('Training Vect Examples')
        #for (count, feature) in zip(test[0].toarray()[0], vectorizer.get_feature_names()):
        #    print(str(count) + ' ' + feature)
        #print(zip(test[0], vectorizer.get_feature_names()))
        #print(test[1])

        return text_clf

    def GetGenericArray(self, x):
        return np.array([t for t in x]).reshape(-1, 1)

    def GetMultipleGenericArray(self, x):
        return x

    def GetIntArray(self, x):
        return x.astype(int)

    def CrossValidationTest(self, X, Y, vectorizer_type, classifier_type):
        kf = model_selection.KFold(n_splits=10)

        female_true = 0
        female_false = 0
        male_true = 0
        male_false = 0

        for train_index, test_index in kf.split(X):
            X = np.array(X, dtype=object)
            Y = np.array(Y, dtype=object)
            X_train, X_test = list(X[train_index]), list(X[test_index])
            Y_train, Y_test = list(Y[train_index]), list(Y[test_index])

            clf = self.BuildClassifier(X_train, Y_train, vectorizer_type, classifier_type)

            conf_matrix = confusion_matrix(Y_test, clf.predict(X_test))
            female_true += conf_matrix[0][0]
            female_false += conf_matrix[1][0]
            male_true += conf_matrix[1][1]
            male_false += conf_matrix[0][1]

        female_accuracy = female_true / (female_false)
        male_accuracy = male_true / (male_false)
        #print("female accuracy", female_accuracy)
        #print("male accuracy", male_accuracy)

        return female_accuracy, male_accuracy