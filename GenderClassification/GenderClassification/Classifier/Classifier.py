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

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

class Classifier(object):

    def RunClassifier(self, training_data_text, training_data_classification, vectorizer_type, classifier_type):
        vectorizer = None
        reducer = None
        classifier = None

        ## Build and Train Model ######################
        if vectorizer_type == 'count':
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
        elif vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(3, 3))
        elif vectorizer_type == 'hash':
            vectorizer = HashingVectorizer(non_negative=True) # This seems like it isn't good
        else:
            print('Unknown Vectorizer: %s' % (vectorizer_type))
            return

        #reducer = SelectKBest(chi2, k=1000)
        #reducer = VarianceThreshold(threshold=0.01)

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

        #print(vectorizer.get_feature_names())
        #lol = len(vectorizer.get_feature_names())

        return text_clf

    def EFS():
        return

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