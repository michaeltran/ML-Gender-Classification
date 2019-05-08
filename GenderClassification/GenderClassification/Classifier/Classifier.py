import time
import numpy as np
import copy

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neural_network import MLPClassifier
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import model_selection
import random
from numpy.random import choice
from scipy.stats import beta
from scipy.stats import expon

from Classes.ColumnExtractor import ColumnExtractor
from Classes.DenseTransformer import DenseTransformer
from Classes.EFS import EFS
from Classes.FSC import FSC
from Classes.ItemSelector import ItemSelector
from Classes.ItemSelectorTF import ItemSelectorTF
from Helper.DebugPrint import DebugPrint

RANDOMIZER_SEED = 1

class Classifier(object):
    def GetFeatures(self, training_data_dict, training_data_classification, vectorizer_pipeline, classifier, feature_selections):
        start = time.time()
        efs_obj = EFS()

        X = vectorizer_pipeline.fit_transform(training_data_dict, training_data_classification)
        candidate_feature_indexes = efs_obj.EFS(X, training_data_classification, classifier, feature_selections)

        end = time.time()
        DebugPrint("Time Run = %fs" % (end - start))

        return candidate_feature_indexes

    ## Multinomial NB
    # Generally discrete values
    # Bin values if non-discrete
    def BuildClassifierNB(self, training_data_dict, training_data_classification, vocab, word_vocab, nb_type):
        ## Build and Train Model ######################
        if nb_type == 'tf': 
            ## TF - 1-GRAM [CHI, IG] [ALL] (0.716 ACC)
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = MultinomialNB()

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
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

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('reducer', reducer),
                        ('scaler', MaxAbsScaler()),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif nb_type == 'discrete': 
            ## Discrete 2-GRAM [ALL] 0.675 ACC
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = MultinomialNB()

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text_2')),
                        ('vectorizer', text_vectorizer),
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

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
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
        elif nb_type == 'bool': 
            ## Bool - 2-GRAM [CHI, IG] (0.92 CV, 0.697 ACC)
            # Changing features2 to smaller bin or using MaxAbsScaler() does nothing
            # Changed to tokenized_text which lowered accuracy
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False, binary=True)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), lowercase=True, tokenizer=lambda x: x.split(' '), binary=True)

            classifier = MultinomialNB()

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tobool', Binarizer()),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tobool', Binarizer()),
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

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG])
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

        #feats = text_clf.named_steps['features']
        #test = feats.transform(training_data_dict)
        #DebugPrint(test[1])
        #print(test[1].toarray())

        return text_clf

    def BuildClassifierSVM(self, training_data_dict, training_data_classification, vocab, word_vocab, svm_type):
        ## Build and Train Model ######################

        if svm_type == 'svmlight-tf':
            # Reducer lowers acc
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = LinearSVC(max_iter=1000000)
            #parameters = [
            #        {'C': [0.1, 1, 2, 3, 4, 5]},
            #    ]
            #final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
                    ])),
                ])

            features2 = FeatureUnion([
                    #('wordcount', Pipeline([
                    #    ('selector', ItemSelector(key='wordcount')),
                    #    ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    #    ('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                    #])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    #('pipeline2', Pipeline([
                    #    ('features', features2),
                    #])),
                ])),
                ('clf', classifier),
            ])
        elif svm_type == 'bool': 
            # Bool - 0.67
            # Kind of sucks
            # Reducer sux
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), lowercase=True)

            classifier = LinearSVC(max_iter=100000, C=0.001, penalty='l2', loss='squared_hinge')
            parameters = [
                    {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge']},
                    {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False]}
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('binarize', Binarizer()),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('binarize', Binarizer()),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        #('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                        ('binarize', Binarizer(threshold=50)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                    ])),
                ])),
                ('clf', final_classifier),
            ])
        elif svm_type == 'tf': 
            # TF 1-GRAM [ALL] - 0.74 ACC
            # 2/3-gram lowers accuracy
            # [CHI, IG] lower accuracy than [ALL] TO DO: TEST
            # Higher accuracy without POS/GPF/FA TO DO: TEST
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' ')) # was 1,2,3

            #classifier = LinearSVC(max_iter=100000, C=1, loss='hinge', penalty='l2')
            classifier = LinearSVC(max_iter=100000, C=10, loss='hinge', penalty='l2')
            parameters = [
                    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge']},
                    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False]}
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
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

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
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
        elif svm_type == 'discrete': 
            # Discrete - 2GRAM 0.70 ACC NO REDUC
            # Normalizer Performs Better Than MinMaxScaler
            # L2 norm performs better than L1 normalization
            # 1-Gram vs 2-Gram doen't matter - TO DO
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = LinearSVC(max_iter=100000)
            parameters = [
                    {'C': [0.1, 1, 2, 3, 4, 5]},
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                    ])),
                ])),
                ('clf', final_classifier),
            ])
        elif svm_type == 'svc': 
            # SVC 0.73 linear 1-gram
            # No reducer
            # Linear and Sigmoid are Good
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = SVC(C=1, kernel='linear')
            parameters = [
                    {'kernel': ['linear'], 'C': [0.1, 1, 2]},
                    #{'kernel': ['rbf'], 'C': [0.1, 1, 2], 'gamma': ['auto', 'scale']},
                    ##{'kernel': ['poly'], 'C': [0.1, 1, 2], 'gamma': ['auto', 'scale'], 'degree': [2, 3, 4]},
                    #{'kernel': ['sigmoid'], 'C': [0.1, 1, 2], 'gamma': ['auto', 'scale']},
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
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

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB(), [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
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
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', final_classifier),
            ])
        elif svm_type == 'usl': 
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))
            text_vectorizer_2 = CountVectorizer(analyzer='word', ngram_range=(2, 3), lowercase=True, tokenizer=lambda x: x.split(' '))

            #classifier = LinearSVC(max_iter=100000, C=1, loss='hinge', penalty='l2')
            classifier = LinearSVC(max_iter=100000, C=10, loss='hinge', penalty='l2')
            classifier = CalibratedClassifierCV(classifier, cv=5)
            #parameters = [
            #        {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge']},
            #        {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False]}
            #    ]
            #final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('text2', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text_2')),
                        ('vectorizer', text_vectorizer_2),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                        ('reducer', SelectPercentile(chi2, 50)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
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
                    ('le_c', Pipeline([
                        ('selector', ItemSelector(key='le_c')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('words_misspelled', Pipeline([
                        ('selector', ItemSelector(key='words_misspelled')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('ts', Pipeline([
                        ('selector', ItemSelector(key='ts')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB(), [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif svm_type == 'usl-b': 
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' ')) # was 1,2,3

            #classifier = LinearSVC(max_iter=100000, C=1, loss='hinge', penalty='l2')
            classifier = LinearSVC(max_iter=100000, C=10, loss='hinge', penalty='l2')
            classifier = CalibratedClassifierCV(classifier, cv=5)
            #parameters = [
            #        {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge']},
            #        {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False]}
            #    ]
            #final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('binarize', Binarizer()),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('binarize', Binarizer()),
                        #('reducer', SelectPercentile(chi2, 50)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        #('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                        ('binarize', Binarizer(threshold=50)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                    ])),
                ])),
                ('clf', classifier),
            ])

        text_clf.fit(training_data_dict, training_data_classification)
        #print(final_classifier.best_params_)
        ###############################################

        return text_clf

    def BuildClassifierSVMR(self, training_data_dict, training_data_classification, vocab, word_vocab, svmr_type):
        ## Build and Train Model ######################
        
        if svmr_type == 'linearsvr':
            # LinearSVR - 0.72
            # L2 > L1
            # TFIDF > TF
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), lowercase=True)

            classifier = LinearSVR(max_iter=100000)
            parameters = [
                    {'C': [0.1, 1, 10], 'epsilon': [0, 0.1, 1], 'loss':('epsilon_insensitive', 'squared_epsilon_insensitive')}
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tf', TfidfTransformer(norm='l2', use_idf=True)),
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

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB(), [FSC.CHI])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', final_classifier),
            ])
        elif svmr_type == 'bool': 
            # Bool - 0.68 1-GRAM
            # RBF
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False, binary=True)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), lowercase=True, binary=True)

            classifier = SVR()
            parameters = [
                    #{'kernel': ['linear'], 'C': [0.1, 1, 10]},
                    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['auto', 'scale']},
                    #{'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': ['auto', 'scale']},
                    #{'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['auto', 'scale']}
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tobool', Binarizer()),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tobool', Binarizer()),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB(), [FSC.CHI, FSC.IG, FSC.MI])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                    ])),
                ])),
                ('clf', final_classifier),
            ])
        elif svmr_type == 'tf': 
            # Term Frequency (Scaled) - 0.68
            # L1 > L2, IDF > !IDF
            # Linear
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), lowercase=True)

            classifier = SVR(kernel='rbf', gamma='scale', C=1, epsilon=0.1)
            parameters = [
                    #{'kernel': ['linear'], 'C': [0.1, 1, 10]},
                    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['auto', 'scale']},
                    #{'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': ['auto', 'scale']},
                    #{'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['auto', 'scale']}
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tf', TfidfTransformer(norm='l1', use_idf=True)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tf', TfidfTransformer(norm='l1', use_idf=True)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tf', TfidfTransformer(norm='l1', use_idf=True)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tf', TfidfTransformer(norm='l1', use_idf=True)),
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

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB(), [FSC.CHI])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', Normalizer(norm='l1')),
                    ])),
                ])),
                ('clf', final_classifier),
            ])

        text_clf.fit(training_data_dict, training_data_classification)
        #print(final_classifier.best_params_)
        ###############################################

        return text_clf

    def BuildClassifierEnsemble(self, training_data_dict, training_data_classification, vocab, word_vocab, ensemble_type):
        ## Build and Train Model ######################
        if ensemble_type == 'bool-bagging':
            # NB
            # 0.68 n=10
            # 0.69 n=1000
            # DT
            # 0.61 n=10
            # 0.71 n=100
            #  n=1000
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))
            
            classifier = BaggingClassifier(base_estimator=None, n_estimators=100, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=7, verbose=0)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('binarize', Binarizer()),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('binarize', Binarizer()),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        #('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                        ('binarize', Binarizer(threshold=50)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif ensemble_type == 'bool-bagging-r':
            # NB
            # 0.68 n=10
            # 0.67 n=100
            # 0.67 n=1000
            # Decision Tree
            # 0.65 n=10
            # 0.68 n=100
            # 0.69 n=1000
            # SVM-R
            # 0.59 n=10
            # 0.59 n=100
            # SVM
            # 0.62 n=10
            # 0.62 n=100
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))
            
            classifier = BaggingRegressor(base_estimator=None, n_estimators=100, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=7, verbose=0)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('binarize', Binarizer()),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('binarize', Binarizer()),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        #('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                        ('binarize', Binarizer(threshold=50)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif ensemble_type == 'discrete-bagging-r':
            # NB
            # DT
            # 0.68 n=100
            # 0.69 n=1000
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))
            
            classifier = BaggingRegressor(base_estimator=None, n_estimators=100, max_features=1.0, max_samples=1.0, 
                                          bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=7, verbose=0)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        #('scaler', Normalizer(norm='l2')),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
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
                        #('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        #('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                    ])),
                ])),
                ('clf', classifier),
            ])
        text_clf.fit(training_data_dict, training_data_classification)
        ###############################################

        return text_clf

    def BuildClassifierKeras(self, training_data_dict, testing_data_dict, vocab, type):
        pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
        text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))
        text_vectorizer_2 = CountVectorizer(analyzer='word', ngram_range=(2, 3), lowercase=True, tokenizer=lambda x: x.split(' '))

        features1 = FeatureUnion([
                ('pos', Pipeline([
                    ('selector', ItemSelector(key='pos')),
                    ('vectorizer', pos_vectorizer),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('text', Pipeline([
                    ('selector', ItemSelector(key='tokenized_text')),
                    ('vectorizer', text_vectorizer),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('text2', Pipeline([
                    ('selector', ItemSelector(key='tokenized_text_2')),
                    ('vectorizer', text_vectorizer_2),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('reducer', SelectPercentile(chi2, 50)),
                ])),
                ('gpf', Pipeline([
                    ('selector', ItemSelector(key='gpf')),
                    ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('fa', Pipeline([
                    ('selector', ItemSelector(key='fa')),
                    ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
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
                    ('le_c', Pipeline([
                        ('selector', ItemSelector(key='le_c')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('words_misspelled', Pipeline([
                        ('selector', ItemSelector(key='words_misspelled')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('ts', Pipeline([
                        ('selector', ItemSelector(key='ts')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('pipeline', Pipeline([
                    ('features', features1),
                    #('scaler', Normalizer()),
                ])),
                ('pipeline2', Pipeline([
                    ('features', features2),
                    ('scaler', MinMaxScaler()),
                    #('scaler', Normalizer()),
                ])),
            ])),
        ])

        text_clf.fit(training_data_dict, training_data_dict['classification'])

        X_train = text_clf.transform(training_data_dict)
        X_test = text_clf.transform(testing_data_dict)
        Y_train = training_data_dict['classification']
        Y_train = np.array(Y_train)
        Y_train[Y_train < 0] = 0
        Y_train = Y_train.tolist()
        Y_test = testing_data_dict['classification']
        Y_test = np.array(Y_test)
        Y_test[Y_test < 0] = 0
        Y_test = Y_test.tolist()

        input_dim = X_train.shape[1]

        model = Sequential()
        # 73.87
        #model.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
        #model.add(layers.Dense(1, activation='sigmoid'))

        # 73.55
        #model.add(layers.Dense(512, input_dim=input_dim, activation='relu'))
        #model.add(layers.Dropout(0.3))
        #model.add(layers.Dense(256, activation='relu'))
        #model.add(layers.Dropout(0.3))
        #model.add(layers.Dense(1, activation='sigmoid'))

        model.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(25, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        #model.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
        #model.add(layers.Dense(25, activation='relu'))
        #model.add(layers.Dense(50, activation='relu'))
        #model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        es = EarlyStopping(monitor='val_acc', mode='auto', verbose=0, restore_best_weights=True, patience=7)

        history = model.fit(X_train, Y_train, epochs=40, verbose=1, validation_data=(X_test, Y_test), batch_size=10, callbacks=[es])

        loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        predictions = model.predict_classes(X_test)

        return history, predictions

    def BuildClassifierMLP(self, training_data_dict, vocab, type):
        pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
        text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))
        text_vectorizer_2 = CountVectorizer(analyzer='word', ngram_range=(2, 3), lowercase=True, tokenizer=lambda x: x.split(' '))
        #text_vectorizer_3 = CountVectorizer(analyzer='char', ngram_range=(2, 3), lowercase=True)

        classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', batch_size='auto', learning_rate='constant', verbose=True, 
                                   early_stopping=True, tol=1e-4, n_iter_no_change=10)

        features1 = FeatureUnion([
                ('pos', Pipeline([
                    ('selector', ItemSelector(key='pos')),
                    ('vectorizer', pos_vectorizer),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('text', Pipeline([
                    ('selector', ItemSelector(key='tokenized_text')),
                    ('vectorizer', text_vectorizer),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('text2', Pipeline([
                    ('selector', ItemSelector(key='tokenized_text_2')),
                    ('vectorizer', text_vectorizer_2),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('reducer', SelectPercentile(chi2, 50)),
                ])),
                #('text3', Pipeline([
                #    ('selector', ItemSelector(key='text')),
                #    ('vectorizer', text_vectorizer_3),
                #    ('tfidf', TfidfTransformer(use_idf=True)),
                #])),
                ('gpf', Pipeline([
                    ('selector', ItemSelector(key='gpf')),
                    ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('fa', Pipeline([
                    ('selector', ItemSelector(key='fa')),
                    ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
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
                    ('le_c', Pipeline([
                        ('selector', ItemSelector(key='le_c')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('words_misspelled', Pipeline([
                        ('selector', ItemSelector(key='words_misspelled')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('ts', Pipeline([
                        ('selector', ItemSelector(key='ts')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('pipeline', Pipeline([
                    ('features', features1),
                    #('scaler', Normalizer()),
                ])),
                ('pipeline2', Pipeline([
                    ('features', features2),
                    ('scaler', MinMaxScaler()),
                    #('scaler', Normalizer()),
                ])),
            ])),
            ('clf', classifier),
        ])

        text_clf.fit(training_data_dict, training_data_dict['classification'])

        return text_clf

    def BuildClassifierSGD(self, training_data_dict, vocab, type):
        pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
        text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), lowercase=True)
        text_vectorizer_2 = CountVectorizer(analyzer='word', ngram_range=(2, 3), lowercase=True, tokenizer=lambda x: x.split(' '))

        classifier = SGDClassifier(loss='hinge', penalty='l2', max_iter=100000, tol=None, n_jobs=4)

        features1 = FeatureUnion([
                ('pos', Pipeline([
                    ('selector', ItemSelector(key='pos')),
                    ('vectorizer', pos_vectorizer),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('text', Pipeline([
                    ('selector', ItemSelector(key='tokenized_text')),
                    ('vectorizer', text_vectorizer),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('text2', Pipeline([
                    ('selector', ItemSelector(key='tokenized_text_2')),
                    ('vectorizer', text_vectorizer_2),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('reducer', SelectPercentile(chi2, 50)),
                ])),
                ('gpf', Pipeline([
                    ('selector', ItemSelector(key='gpf')),
                    ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('fa', Pipeline([
                    ('selector', ItemSelector(key='fa')),
                    ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
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
                    ('le_c', Pipeline([
                        ('selector', ItemSelector(key='le_c')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('words_misspelled', Pipeline([
                        ('selector', ItemSelector(key='words_misspelled')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('ts', Pipeline([
                        ('selector', ItemSelector(key='ts')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('pipeline', Pipeline([
                    ('features', features1),
                ])),
                ('pipeline2', Pipeline([
                    ('features', features2),
                    ('scaler', MinMaxScaler()),
                ])),
            ])),
            ('clf', classifier),
        ])

        text_clf.fit(training_data_dict, training_data_dict['classification'])

        return text_clf

    def SemiSupervisedLearning(self, training_data_dict, testing_data_dict, blog_data_dict, vocab):
        training = copy.deepcopy(training_data_dict)
        blog = copy.deepcopy(blog_data_dict)

        MAX_ITERATIONS = 1
        MAX_ADDED = 200

        iteration = 0
        previous_training_length = 0
        while len(training['index']) != previous_training_length and iteration < MAX_ITERATIONS:
            iteration += 1
            previous_training_length = len(training['index'])

            kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=RANDOMIZER_SEED + iteration - 1)
            for train_index, test_index in kf.split(training_data_dict['index']):
                X_train = {}
                Y_test = {}
                for key in training_data_dict:
                    X_train[key] = [training_data_dict[key][i] for i in train_index]
                for key in training_data_dict:
                    Y_test[key] = [training_data_dict[key][i] for i in test_index]


                clf_tst = self.BuildClassifierSVM(training, training['classification'], vocab, None, 'usl')
                predictions = clf_tst.predict(testing_data_dict)
                print("%d - Accuracy: %0.3f" % (iteration, accuracy_score(testing_data_dict['classification'], predictions)))

                clf = self.BuildClassifierSVM(Y_test, Y_test['classification'], vocab, None, 'usl')
                #predictions = clf.predict(testing_data_dict)
                #print("%d - Accuracy: %0.3f" % (iteration, accuracy_score(testing_data_dict['classification'], predictions)))

                probability_list = clf.predict_proba(blog)
                predictions = clf.predict(blog)
                #print(clf.classes_)
                # [-1, 1]

                indexes_male = np.argsort(-probability_list[:,1])
                indexes_female = np.argsort(-probability_list[:,0])

                #print(probability_list[indexes_male[0]])
                #print(probability_list[indexes_male[1]])

                #print(probability_list[indexes_female[0]])
                #print(probability_list[indexes_female[1]])

                to_remove = []

                #distribution = beta.rvs(100, 1, size=len(probability_list))
                #distribution = distribution/(np.sum(distribution))

                distribution = expon.rvs(scale=1, loc=0, size=len(probability_list))
                distribution = -np.sort(-distribution)
                distribution = distribution/(np.sum(distribution))

                indexes_male_picked = choice(indexes_male, len(indexes_male), p=distribution)
                indexes_female_picked = choice(indexes_female, len(indexes_female), p=distribution)

                males_added = 0
                i = 0
                while males_added < MAX_ADDED:
                    #index = indexes_male[i]
                    index = indexes_male_picked[i]
                    if predictions[index] == blog['classification'][index] and index not in to_remove:
                        males_added += 1
                    
                        to_remove.append(index)
                        for key in training:
                            training[key].append(blog[key][index])
                    i += 1

                females_added = 0
                i = 0
                while females_added < MAX_ADDED:
                    #index = indexes_female[i]
                    index = indexes_female_picked[i]
                    if predictions[index] == blog['classification'][index] and index not in to_remove:
                        females_added += 1
                    
                        to_remove.append(index)
                        for key in training:
                            training[key].append(blog[key][index])
                    i += 1

                to_remove.sort()
                #to_remove = list(dict.fromkeys(to_remove))
                for i in range(len(to_remove)):
                    index = to_remove[i] - i
                    for key in blog:
                        del blog[key][index]
        return training

    def SemiSupervisedLearning2(self, training_data_dict, testing_data_dict, blog_data_dict, vocab):
        training = dict(training_data_dict)
        blog = dict(blog_data_dict)
        added_blog = {}

        for key in training:
            added_blog[key] = []

        for i in range(len(testing_data_dict['index'])):
            for key in training:
                training[key].append(testing_data_dict[key][i])

        MAX_ITERATIONS = 2
        MAX_ADDED = 200

        iteration = 0
        previous_training_length = 0
        while len(training['index']) != previous_training_length and iteration < MAX_ITERATIONS:
            iteration += 1
            previous_training_length = len(training['index'])
            clf = self.BuildClassifierSVM(training, training['classification'], vocab, None, 'usl')
            #predictions = clf.predict(testing_data_dict)
            #print("%d - Accuracy: %0.3f" % (iteration, accuracy_score(testing_data_dict['classification'], predictions)))

            males_added = 0
            females_added = 0

            probability_list = clf.predict_proba(blog)
            print(clf.classes_)
            # [-1, 1]
            indexes_male = np.argsort(-probability_list[:,1])
            indexes_female = np.argsort(-probability_list[:,0])
            predictions = clf.predict(blog)

            print(probability_list[indexes_male[0]])
            print(probability_list[indexes_male[1]])

            print(probability_list[indexes_female[0]])
            print(probability_list[indexes_female[1]])

            to_remove = []

            i = 0
            while males_added < MAX_ADDED:
                if predictions[indexes_male[i]] == blog['classification'][indexes_male[i]]:
                    males_added += 1
                    
                    to_remove.append(indexes_male[i])
                    for key in training:
                        training[key].append(blog[key][indexes_male[i]])
                        added_blog[key].append(blog[key][indexes_male[i]])
                i += 1
            i = 0
            while females_added < MAX_ADDED:
                if predictions[indexes_female[i]] == blog['classification'][indexes_female[i]]:
                    females_added += 1
                    
                    to_remove.append(indexes_female[i])
                    for key in training:
                        training[key].append(blog[key][indexes_female[i]])
                        added_blog[key].append(blog[key][indexes_female[i]])
                i += 1

            to_remove.sort()
            for i in range(len(to_remove)):
                for key in blog:
                    del blog[key][to_remove[i]-i]
        return added_blog

    def GetGenericArray(self, x):
        return np.array([t for t in x]).reshape(-1, 1)

    def GetMultipleGenericArray(self, x):
        return x