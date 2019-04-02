import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from Classes.ColumnExtractor import ColumnExtractor
from Classes.DenseTransformer import DenseTransformer
from Classes.EFS import EFS
from Classes.ItemSelector import ItemSelector
from Classes.ItemSelectorTF import ItemSelectorTF

import time

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

    def GetFeatures(self, training_data_dict, training_data_classification, vectorizer_pipeline, classifier):
        #start = time.time()

        X = vectorizer_pipeline.fit_transform(training_data_dict, training_data_classification)
        efs_obj = EFS()
        candidate_feature_indexes = efs_obj.EFS(X, training_data_classification, classifier)

        #end = time.time()
        #print("Time Run = %fs" % (end - start))

        return candidate_feature_indexes

    ## Multinomial NB
    # Generally discrete values
    # Bin values if non-discrete
    def BuildClassifierNB(self, training_data_dict, training_data_classification, vocab, nb_type):
        ## Build and Train Model ######################

        #nb_type = 'tf'
        #nb_type = 'discrete'
        #nb_type = 'bool'

        if 'tf': # TF - 1-GRAM [CHI, IG] (0.72 CV, 0.69 ACC)
            vectorizer1 = TfidfVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False, use_idf=True)
            vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '), use_idf=True)

            classifier = MultinomialNB()

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', vectorizer1),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', vectorizer2),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelectorTF(key='gpf', keycount='wordcount')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelectorTF(key='fa', keycount='wordcount')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                ])

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier)
            reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif nb_type == 'discrete': # Discrete Counts 0.69 ACC
            vectorizer1 = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False)
            vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = MultinomialNB()

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', vectorizer1),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', vectorizer2),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                ])

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier)
            reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        #('scaler', StandardScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif nb_type == 'bool': # Bool - 2-GRAM [CHI, IG] (0.92 CV, 0.69 ACC)
            vectorizer1 = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False, binary=True)
            vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2), lowercase=True, tokenizer=lambda x: x.split(' '), binary=True)

            classifier = MultinomialNB()

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', vectorizer1),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', vectorizer2),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tobool', FunctionTransformer(self.GetBoolArray, validate = False)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tobool', FunctionTransformer(self.GetBoolArray, validate = False)),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                ])

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier)
            reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
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

        feats = text_clf.named_steps['features']
        test = feats.transform(training_data_dict)
        print(test[1])

        return text_clf

    def BuildClassifierSVM(self, training_data_dict, training_data_classification, vocab, svm_type):

        #svm_type = 'tf'
        #svm_type = 'discrete'
        #svm_type = 'svc'

        if svm_type == 'tf': # TF [CHI, IG] - 0.79 CV, 0.70 ACC
            vectorizer1 = TfidfVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False, use_idf=True)
            vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '), use_idf=True)

            classifier = LinearSVC()

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', vectorizer1),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', vectorizer2),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelectorTF(key='gpf', keycount='wordcount')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelectorTF(key='fa', keycount='wordcount')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier)
            reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif svm_type == 'discrete': # Discrete - 0.65 CV, 0.64 ACC, 2GRAM 0.69 ACC [CHI, IG]

            # TO DO
            vectorizer1 = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False)
            vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = LinearSVC(max_iter=10000)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', vectorizer1),
                        ('dense', DenseTransformer()),
                        ('tofloat', FunctionTransformer(self.GetFloatArray, validate = False)),
                        ('scaler', MinMaxScaler()),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', vectorizer2),
                        ('dense', DenseTransformer()),
                        ('tofloat', FunctionTransformer(self.GetFloatArray, validate = False)),
                        ('scaler', MinMaxScaler()),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('scaler', MinMaxScaler()),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier)
            reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif svm_type == 'svc': # SVC 0.68 linear
            vectorizer1 = TfidfVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False, use_idf=False)
            vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '), use_idf=False)

            classifier = SVC(C=1, kernel='linear', degree=3, gamma='auto')

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', vectorizer1),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', vectorizer2),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelectorTF(key='gpf', keycount='wordcount')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelectorTF(key='fa', keycount='wordcount')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier)
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                        ('scaler', MaxAbsScaler()),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MaxAbsScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])

        text_clf.fit(training_data_dict, training_data_classification)
        ###############################################

        feats = text_clf.named_steps['features']
        test = feats.transform(training_data_dict)
        print(test[1])

        return text_clf

    def BuildClassifierSVMR(self, training_data_dict, training_data_classification, vocab, svmr_type):
        vectorizer = None
        reducer = None
        classifier = None

        #svmr_type = 'bool'
        #svmr_type = 'tf'

        ## Build and Train Model ######################
        if svmr_type == 'bool': # Bool
            vectorizer1 = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False, binary=True)
            vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 3), lowercase=True, binary=True)

            classifier = SVR(kernel='rbf', gamma='auto', C=1, epsilon=0.1)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', vectorizer1),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', vectorizer2),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tobool', FunctionTransformer(self.GetBoolArray, validate = False)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tobool', FunctionTransformer(self.GetBoolArray, validate = False)),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB())
            reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', StandardScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif svmr_type == 'tf': # Term Frequency (Scaled)
            vectorizer1 = TfidfVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False, use_idf=False)
            vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True)

            classifier = SVR(kernel='rbf', gamma='auto', C=1, epsilon=0.1)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', vectorizer1),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', vectorizer2),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelectorTF(key='gpf', keycount='wordcount')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelectorTF(key='fa', keycount='wordcount')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ])),
                ])

            features2 = FeatureUnion([
                    #('length', Pipeline([
                    #    ('selector', ItemSelector(key='length')),
                    #    ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    #])),
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB())
            reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('reducer', reducer),
                        ('dense', DenseTransformer()),
                        ('scaler', StandardScaler()),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', StandardScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])

        text_clf.fit(training_data_dict, training_data_classification)
        ###############################################

        feats = text_clf.named_steps['features']
        test = feats.transform(training_data_dict)
        print(test[1])

        return text_clf

    def GetGenericArray(self, x):
        return np.array([t for t in x]).reshape(-1, 1)

    def GetMultipleGenericArray(self, x):
        return x

    def GetBoolArray(self, x):
        return np.where(np.array(x) > 0, 1, 0)

    def GetIntArray(self, x):
        return x.astype(int)

    def GetFloatArray(self, x):
        return x.astype(float)