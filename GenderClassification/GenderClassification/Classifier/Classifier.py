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
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import cross_val_score

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

class Classifier(object):

    def GetFeatures(self, training_data_text, training_data_classification, vectorizer_type):
        if vectorizer_type == 'count':
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        elif vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
        elif vectorizer_type == 'hash':
            vectorizer = HashingVectorizer(non_negative=True) # This seems like it isn't good
        else:
            print('Unknown Vectorizer: %s' % (vectorizer_type))
            return

        X = vectorizer.fit_transform(training_data_text, training_data_classification)
        candidate_feature_indexes = self.EFS(X.toarray(), training_data_classification)

        return candidate_feature_indexes

    def RunClassifier(self, training_data_text, training_data_classification, vectorizer_type, classifier_type, features):
        vectorizer = None
        reducer = None
        classifier = None

        ## Build and Train Model ######################
        if vectorizer_type == 'count':
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        elif vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
        elif vectorizer_type == 'hash':
            vectorizer = HashingVectorizer(non_negative=True) # This seems like it isn't good
        else:
            print('Unknown Vectorizer: %s' % (vectorizer_type))
            return

        #reducer = SelectKBest(chi2, k=5000)
        #reducer = VarianceThreshold(threshold=0.01)
        #reducer = SelectKBest(self.EFS, k=1000)
        reducer = ColumnExtractor(cols=features)

        if classifier_type == 'nb':
            classifier = MultinomialNB()
        elif classifier_type == 'svm':
            classifier = SVC(kernel='linear')
        elif classifier_type == 'dt':
            classifier = DecisionTreeClassifier()
        elif classifier_type == 'rf':
            classifier = RandomForestClassifier(n_estimators=10)
        elif classifier_type == 'log':
            classifier = LogisticRegression(C=.088)
        else:
            print('Unknown Classifier: %s' % (classifier_type))
            return

        text_clf = Pipeline([
                ('vect', vectorizer),
                ('reduce_dim', reducer),
                ('clf', classifier)
            ])

        text_clf.fit(training_data_text, training_data_classification)
        ###############################################

        #features = text_clf.named_steps['reduce_dim']
        #feature_names = vectorizer.get_feature_names()
        #mask = features.get_support()
        #print(np.extract(mask, feature_names))

        return text_clf

    def RunStandardClassifier(self, training_data_text, training_data_classification, classifier_type):
        vectorizer = None
        reducer = None
        classifier = None

        ## Build and Train Model ######################
        if classifier_type == 'nb':
            classifier = MultinomialNB()
        elif classifier_type == 'svm':
            classifier = SVC(kernel='linear')
        elif classifier_type == 'dt':
            classifier = DecisionTreeClassifier()
        elif classifier_type == 'rf':
            classifier = RandomForestClassifier(n_estimators=10)
        elif classifier_type == 'log':
            classifier = LogisticRegression(C=.088)
        else:
            print('Unknown Classifier: %s' % (classifier_type))
            return

        text_clf = Pipeline([
                ('clf', classifier)
            ])
        text_clf.fit(training_data_text, training_data_classification)
        ###############################################

        return text_clf

    def InformationGain(self, X, Y):
        IG = []
        C = Y.count(1)
        C_ = Y.count(0)
        N = C + C_
        P_c = C / N
        P_c_ = 1 - P_c

        for i in range(X.shape[1]):
            X_When_Y_1 = np.extract(Y, X[:,i])
            temp = np.where(X_When_Y_1 > 0)
            w = float(temp[0].size)

            temp = np.where(X[:,i] > 0)
            x = float(temp[0].size - w)

            temp = np.where(X_When_Y_1 == 0)
            y = float(temp[0].size)

            temp = np.where(X[:,i] == 0)
            z = float(temp[0].size - y)

            P_f = (w + x) / N
            P_f_ = 1 - P_f

            IG_c = P_c * np.log2(P_c) + P_c_ * np.log2(P_c_)

            IG_f_1 = 0
            IG_f_2 = 0

            if w != 0:
                IG_f_1 = (w / (w + x)) * np.log2((w / (w + x)))
            if x != 0:
                IG_f_2 = (x / (w + x)) * np.log2((x / (w + x)))
            IG_f = P_f * ( IG_f_1 + IG_f_2 )

            IG_f_ = P_f_ * ( (y / (y + z)) * np.log2((y / (y + z))) + (z / (y + z)) * np.log2((z / (y + z))) )

            IG_ = -IG_c + (IG_f + IG_f_)
            IG.append(IG_)
        return IG

    def MutualInformation(self, X, Y):
        MI = []
        C = Y.count(1)
        C_ = Y.count(0)
        N = C + C_
        P_c = C / N
        P_c_ = 1 - P_c

        for i in range(X.shape[1]):
            X_When_Y_1 = np.extract(Y, X[:,i])
            temp = np.where(X_When_Y_1 > 0)
            w = float(temp[0].size)
            temp = np.where(X[:,i] > 0)
            x = float(temp[0].size - w)
            temp = np.where(X_When_Y_1 == 0)
            y = float(temp[0].size)
            temp = np.where(X[:,i] == 0)
            z = float(temp[0].size - y)

            P_fc = w / N
            P_fc_ = x / N
            P_f_c = y / N
            P_f_c_ = z / N
            P_f = (w + x) / N
            P_f_ = 1 - P_f

            MI1 = 0
            MI2 = 0
            MI3 = 0
            MI4 = 0

            if P_fc != 0:
                MI1 = P_fc * np.log2(P_fc / (P_f * P_c))
            if P_fc_ != 0:
                MI2 = P_fc_ * np.log2(P_fc_ / (P_f * P_c_))
            if P_f_c != 0:
                MI3 = P_f_c * np.log2(P_f_c / (P_f_ * P_c))
            if P_f_c_ != 0:
                MI4 = P_f_c_ * np.log2(P_f_c_ / (P_f_ * P_c_))

            MI_f = MI1 + MI2 + MI3 + MI4

            MI.append(MI_f)
        return MI

    def CrossEntropy(self, X, Y):
        CE = []
        C = Y.count(1)
        C_ = Y.count(0)
        N = C + C_
        P_c = C / N
        P_c_ = 1 - P_c

        for i in range(X.shape[1]):
            X_When_Y_1 = np.extract(Y, X[:,i])
            temp = np.where(X_When_Y_1 > 0)
            w = float(temp[0].size)
            temp = np.where(X[:,i] > 0)
            x = float(temp[0].size - w)
            temp = np.where(X_When_Y_1 == 0)
            y = float(temp[0].size)
            temp = np.where(X[:,i] == 0)
            z = float(temp[0].size - y)

            P_f = (w + x) / N

            CE_ = P_f * ((w / (w + x)) * np.log2((w / (w + x)) / P_f) + (x / (w + x)) * np.log2((x / (w + x)) / P_f))

            CE.append(CE_)
        return CE

    def WeightOfEvidenceForText(self, X, Y):
        WET = []
        C = Y.count(1)
        C_ = Y.count(0)
        N = C + C_
        P_c = C / N
        P_c_ = 1 - P_c

        for i in range(X.shape[1]):
            X_When_Y_1 = np.extract(Y, X[:,i])
            temp = np.where(X_When_Y_1 > 0)
            w = float(temp[0].size)
            temp = np.where(X[:,i] > 0)
            x = float(temp[0].size - w)
            temp = np.where(X_When_Y_1 == 0)
            y = float(temp[0].size)
            temp = np.where(X[:,i] == 0)
            z = float(temp[0].size - y)

            P_f = (w + x) / N
            P_f_ = 1 - P_f

            P_cGf = w / (w + x)
            P_c_Gf = x / (w + x)

            WET1 = P_c * P_f * abs(np.log2( (P_cGf * (1 - P_c)) / (P_c * (1 - P_cGf)) ))
            WET2 = P_c_ * P_f * abs(np.log2( (P_c_Gf * (1 - P_c_)) / (P_c_ * (1 - P_c_Gf)) ))

            WET_ = WET1 + WET2

            WET.append(WET_)
        return WET

    def EFS(self, X, Y):
        t = 3 # number of feature selection algorithms
        w = int(X.shape[1]/100) # window size
        tau_i = int(X.shape[1]/20) # tau
        step_size = int(w / 5)

        #test = self.InformationGain(X, Y)

        Xi = []
        Xi.append(chi2(X, Y)[0])
        Xi.append(self.InformationGain(X, Y))
        Xi.append(self.MutualInformation(X, Y))
        #Xi.append(self.CrossEntropy(X, Y))
        #Xi.append(self.WeightOfEvidenceForText(X, Y))

        #Xi.append(mutual_info_classif(X, Y, discrete_features=True))

        C = []
        for i in range(0, t):
            C_i = []
            iterator = 0
            for tau in range(tau_i-w, tau_i+w):
                if (iterator % step_size == 0):
                    zeta_i = np.argsort(Xi[i])[-tau:]
                    C_i.append(zeta_i)
                iterator += 1
            C.append(C_i)

        OptCandFeatures = []
        while len(C[0]) > 0:
            big_lambda = []
            for i in range(0, t):
                first_feature_set = C[i][0]
                del C[i][0]
                big_lambda = np.union1d(big_lambda, first_feature_set).astype(int)
            OptCandFeatures.append(big_lambda)

        scores = []
        for i in range(len(OptCandFeatures)):
            candidate_feature_indexes = OptCandFeatures[i]
            candidate_features = X[:,candidate_feature_indexes]
            text_clf = self.RunStandardClassifier(candidate_features, Y, 'nb')
            cv_scores = cross_val_score(text_clf, candidate_features, Y, cv=10, scoring='accuracy')
            scores.append(cv_scores.mean())
            print("%d - Cross Validation Accuracy: %0.4f (+/- %0.2f)" % (i, cv_scores.mean(), cv_scores.std()))

        best_score_index = scores.index(max(scores))
        best_score = scores[best_score_index]

        print("Best Scoring Index: %d" % (best_score_index))

        candidate_feature_indexes = OptCandFeatures[best_score_index]

        return candidate_feature_indexes

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

            clf = self.RunClassifier(X_train, Y_train, vectorizer_type, classifier_type)

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

class ColumnExtractor(object):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        sliced = X[:, self.cols]
        return sliced

    def fit(self, X, y=None):
        return self