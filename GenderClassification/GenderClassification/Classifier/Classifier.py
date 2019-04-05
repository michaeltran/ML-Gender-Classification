import time
import numpy as np

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from Classes.ColumnExtractor import ColumnExtractor
from Classes.DenseTransformer import DenseTransformer
from Classes.EFS import EFS
from Classes.FSC import FSC
from Classes.ItemSelector import ItemSelector
from Classes.ItemSelectorTF import ItemSelectorTF
from Helper.DebugPrint import DebugPrint


class Classifier(object):
    def GetFeatures(self, training_data_dict, training_data_classification, vectorizer_pipeline, classifier, feature_selections):
        start = time.time()

        X = vectorizer_pipeline.fit_transform(training_data_dict, training_data_classification)
        efs_obj = EFS()
        candidate_feature_indexes = efs_obj.EFS(X, training_data_classification, classifier, feature_selections)

        end = time.time()
        DebugPrint("Time Run = %fs" % (end - start))

        return candidate_feature_indexes

    ## Multinomial NB
    # Generally discrete values
    # Bin values if non-discrete
    def BuildClassifierNB(self, training_data_dict, training_data_classification, vocab, word_vocab, nb_type):
        ## Build and Train Model ######################
        if nb_type == 'default':
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), lowercase=True, tokenizer=lambda x: x.split(' '))

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
        elif nb_type == 'tf': 
            ## TF - 1-GRAM [CHI, IG] [ALL] (0.69 ACC)
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
                        ('scaler', MaxAbsScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif nb_type == 'discrete': 
            ## Discrete 2-GRAM [ALL] 0.67 ACC
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), lowercase=True, tokenizer=lambda x: x.split(' '))

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
            ## Bool - 2-GRAM [CHI, IG] (0.92 CV, 0.70 ACC)
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

        feats = text_clf.named_steps['features']
        test = feats.transform(training_data_dict)
        DebugPrint(test[1])
        #print(test[1].toarray())

        return text_clf

    def BuildClassifierSVM(self, training_data_dict, training_data_classification, vocab, word_vocab, svm_type):
        ## Build and Train Model ######################

        if svm_type == 'default':
            pos_vectorizer = TfidfVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False, use_idf=False)
            text_vectorizer = TfidfVectorizer(vocabulary=word_vocab, analyzer='word', ngram_range=(1, 2), lowercase=True, tokenizer=lambda x: x.split(' '), use_idf=False)

            classifier = LinearSVC(max_iter=10000, C=1, penalty='l2', loss='hinge')
            #parameters = [
            #    #{'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False], 'C': [0.1, 1]}, 
            #    {'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'], 'C': [0.1, 1]}]
            #final_class = GridSearchCV(classifier, parameters, cv=5)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', text_vectorizer),
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

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, final_class, [FSC.CHI, FSC.IG])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('pca', TruncatedSVD(n_components=100)),
                        ('poly', PolynomialFeatures(degree=2)),
                        #('reducer', reducer),
                    ])),
                    #('pipeline2', Pipeline([
                    #    ('features', features2),
                    #    ('scaler', MaxAbsScaler()),
                    #])),
                ])),
                ('clf', classifier),
            ])
        elif svm_type == 'tf': # TF [CHI, IG] - 0.70 ACC
            pos_vectorizer = TfidfVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False, use_idf=True)
            text_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '), use_idf=True)

            classifier = LinearSVC()

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', text_vectorizer),
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
                        ('scaler', MaxAbsScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif svm_type == 'discrete': # Discrete - 2GRAM 0.70 ACC NO REDUC
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = LinearSVC(max_iter=10000)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('scaler', MaxAbsScaler()),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', text_vectorizer),
                        ('scaler', MaxAbsScaler()),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('scaler', MaxAbsScaler()),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('scaler', MaxAbsScaler()),
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
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MaxAbsScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif svm_type == 'svc': # SVC 0.68 linear
            pos_vectorizer = TfidfVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False, use_idf=False)
            text_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '), use_idf=False)

            classifier = SVC(C=1, kernel='linear', degree=3, gamma='auto')

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', text_vectorizer),
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
        #print(final_class.best_params_)
        #print(classifier.coef_ )
        ###############################################

        feats = text_clf.named_steps['features']
        test = feats.transform(training_data_dict)
        DebugPrint(test[1])
        #print(test[1])

        return text_clf

    def BuildClassifierSVMR(self, training_data_dict, training_data_classification, vocab, word_vocab, svmr_type):
        ## Build and Train Model ######################
        
        if svmr_type == 'linearsvr':
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(vocabulary=word_vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=True)

            classifier = LinearSVR(max_iter=10000)
            parameters = [
                {'C': [0.01, 0.1, 1, 10], 'epsilon': [0, 0.1, 1], 'loss':('epsilon_insensitive', 'squared_epsilon_insensitive')}
                ]
            final_class = GridSearchCV(classifier, parameters, cv=5)

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
                        ('tf', TfidfTransformer(norm='l1', use_idf=False)),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', final_class),
            ])
        elif svmr_type == 'default':
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False, binary=True)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, binary=True)

            classifier = SVR(kernel='rbf', gamma='auto', C=1, epsilon=0.1)
            parameters = [
                {'kernel': ('linear', 'rbf'), 'C': [0.1, 10]}]
            final_class = GridSearchCV(classifier, parameters, cv=5)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', text_vectorizer),
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

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB(), [FSC.CHI])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('pca', TruncatedSVD(n_components=10)),
                        #('reducer', reducer),
                        #('dense', DenseTransformer()),
                        #('scaler', StandardScaler()),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', StandardScaler()),
                    ])),
                ])),
                ('clf', final_class),
            ])
        elif svmr_type == 'bool': # Bool - 0.65 1-GRAM
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False, binary=True)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, binary=True)

            classifier = SVR(kernel='rbf', gamma='auto', C=1, epsilon=0.1)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', text_vectorizer),
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
                        ('scaler', StandardScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        elif svmr_type == 'tf': # Term Frequency (Scaled) - 0.65
            pos_vectorizer = TfidfVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False, use_idf=False)
            text_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True)

            classifier = SVR(kernel='rbf', gamma='auto', C=1, epsilon=0.1)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vectorizer', text_vectorizer),
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

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB(), [FSC.CHI])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
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
        #print(final_class.best_params_)
        ###############################################

        feats = text_clf.named_steps['features']
        test = feats.transform(training_data_dict)
        DebugPrint(test[1])
        #print(test[1])

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