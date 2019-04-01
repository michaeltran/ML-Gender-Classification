import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

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

from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif

from Classes.ColumnExtractor import ColumnExtractor
from Classes.DenseTransformer import DenseTransformer
from Classes.EFS import EFS
from Classes.ItemSelector import ItemSelector
from Classes.ItemSelectorTF import ItemSelectorTF

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

    def GetFeatures(self, training_data_dict, training_data_classification, vocab):
        #vectorizer = self.GetVectorizer(vectorizer_type)
        vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False, binary=True)

        vectorizer_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('pos', Pipeline([
                    ('selector', ItemSelector(key='pos')),
                    ('vectorizer', vectorizer),
                ])),
            ]))
        ])

        X = vectorizer_pipeline.fit_transform(training_data_dict, training_data_classification)
        candidate_feature_indexes = efs_obj.EFS(X.toarray(), training_data_classification)

        return candidate_feature_indexes

    def GetFeatures2(self, training_data_dict, training_data_classification, vocab):
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))

        vectorizer_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('pos', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('vectorizer', vectorizer),
                ])),
            ]))
        ])

        X = vectorizer_pipeline.fit_transform(training_data_dict, training_data_classification)
        candidate_feature_indexes = efs_obj.EFS(X.toarray(), training_data_classification)

        return candidate_feature_indexes

    def GetFeatures3(self, training_data_dict, training_data_classification, vectorizer_pipeline):
        X = vectorizer_pipeline.fit_transform(training_data_dict, training_data_classification)
        candidate_feature_indexes = efs_obj.EFS(X.toarray(), training_data_classification)

        return candidate_feature_indexes

    ## Multinomial NB
    # Generally discrete values
    # Bin values if non-discrete
    def BuildClassifierNB(self, training_data_dict, training_data_classification, features, vocab):
        ## Build and Train Model ######################
        #vectorizer1 = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False)
        #vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(3, 3), tokenizer=lambda x: x.split(' '), lowercase=False)
        #reducer = ColumnExtractor(cols=features)
        #classifier = MultinomialNB()

        #text_clf = Pipeline([
        #    ('features', FeatureUnion([
        #        ('pos', Pipeline([
        #            ('selector', ItemSelector(key='pos')),
        #            ('vectorizer', vectorizer1),
        #            ('reducer', reducer),
        #        ])),
        #        ('text', Pipeline([
        #            ('selector', ItemSelector(key='text')),
        #            ('vectorizer', vectorizer2),
        #            #('reducer', reducer),
        #        ])),
        #        ('length', Pipeline([
        #            ('selector', ItemSelector(key='length')),
        #            ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
        #            ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
        #            ('toint', FunctionTransformer(self.GetIntArray, validate = False)),
        #        ])),
        #        ('fmeasure', Pipeline([
        #            ('selector', ItemSelector(key='fmeasure')),
        #            ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
        #            ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
        #            ('toint', FunctionTransformer(self.GetIntArray, validate = False)),
        #        ])),
        #        ('gpf', Pipeline([
        #            ('selector', ItemSelector(key='gpf')),
        #            ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
        #        ])),
        #        ('fa', Pipeline([
        #            ('selector', ItemSelector(key='fa')),
        #            ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
        #        ])),
        #    ])),
        #    ('clf', classifier),
        #])

        if True:
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

            reducer_features = self.GetFeatures3(training_data_dict, training_data_classification, features1)
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
        else:
            vectorizer1 = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False, binary=True)
            vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, binary=True)

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

            reducer_features = self.GetFeatures3(training_data_dict, training_data_classification, features1)
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

    def BuildClassifierSVM(self, training_data_dict, training_data_classification, features, vocab, UseTF):
        vectorizer = None
        reducer = None
        classifier = None

        ## Build and Train Model ######################
        if UseTF == False:
            vectorizer1 = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 4), tokenizer=lambda x: x.split(' '), lowercase=False, binary=True)
            vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, binary=True)

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

            reducer_features = self.GetFeatures3(training_data_dict, training_data_classification, features1)
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
        elif UseTF == True:
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

            reducer_features = self.GetFeatures3(training_data_dict, training_data_classification, features1)
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

        #feats = text_clf.named_steps['features']
        #test = feats.transform(training_data_dict)
        #print('Training Vect Examples')
        #for (count, feature) in zip(test[0].toarray()[0], vectorizer.get_feature_names()):
        #    print(str(count) + ' ' + feature)
        #print(zip(test[0], vectorizer.get_feature_names()))
        #print(test[1])

        #pipe = text_clf.named_steps['pipeline']
        #test = pipe.transform(training_data_dict)

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