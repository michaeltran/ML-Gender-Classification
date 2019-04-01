import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score

class EFS(object):
    def GetContingencyTable(self, X, Y, feature):
        #   1 0
        # 1 w x
        # 0 y z

        # X_When_Y_1 = Values of X when Y = 1
        # temp = indexes when feature > 0

        #test = X[:,feature]
        #testtest = test.toarray()

        X_When_Y_1 = np.extract(Y, X[:,feature])
        temp = np.where(X_When_Y_1 > 0)
        w = float(temp[0].size)

        temp = np.where(X[:,feature] > 0)
        x = float(temp[0].size - w)

        temp = np.where(X_When_Y_1 == 0)
        y = float(temp[0].size)

        temp = np.where(X[:,feature] == 0)
        z = float(temp[0].size - y)

        return w, x, y, z

    def InformationGain(self, X, Y):
        IG = []
        C = Y.count(1)
        C_ = Y.count(0)
        N = C + C_
        P_c = C / N
        P_c_ = 1 - P_c

        IG_c = P_c * np.log2(P_c) + P_c_ * np.log2(P_c_)

        for i in range(X.shape[1]):
            w, x, y, z = self.GetContingencyTable(X, Y, i)

            P_f = (w + x) / N
            P_f_ = 1 - P_f

            IG_f_1 = 0
            IG_f_2 = 0

            if w != 0:
                IG_f_1 = (w / (w + x)) * np.log2((w / (w + x)))
            if x != 0:
                IG_f_2 = (x / (w + x)) * np.log2((x / (w + x)))
            IG_f = P_f * ( IG_f_1 + IG_f_2 )

            IG_f_1 = 0
            IG_f_2 = 0

            if y != 0:
                IG_f_1 = (y / (y + z)) * np.log2((y / (y + z)))
            if z != 0:
                IG_f_2 = (z / (y + z)) * np.log2((z / (y + z)))
            IG_f_ = P_f_ * ( IG_f_1 + IG_f_2 )

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
            w, x, y, z = self.GetContingencyTable(X, Y, i)

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

    def ChiSquared(self, X, Y):
        chi_squared = []
        C = Y.count(1)
        C_ = Y.count(0)
        N = C + C_

        for i in range(X.shape[1]):
            w, x, y, z = self.GetContingencyTable(X, Y, i)

            chi2_fc = ( N * np.power(((w * z) - (y * x)), 2) ) / ( (w + y) * (x + z) * (w + x) * (y + z) )

            chi_squared.append(chi2_fc)
        return chi_squared

    def CrossEntropy(self, X, Y):
        CE = []
        C = Y.count(1)
        C_ = Y.count(0)
        N = C + C_
        P_c = C / N
        P_c_ = 1 - P_c

        for i in range(X.shape[1]):
            w, x, y, z = self.GetContingencyTable(X, Y, i)

            P_f = (w + x) / N

            P_cGf = w / (w + x)
            P_c_Gf = x / (w + x)

            CE1 = 0
            CE2 = 0
            if P_cGf != 0:
                CE1 = P_cGf * np.log2(P_cGf / P_f)
            if P_c_Gf != 0:
                CE2 = P_c_Gf * np.log2(P_c_Gf / P_f)
            CE_ = P_f * (CE1 + CE2)

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
            w, x, y, z = self.GetContingencyTable(X, Y, i)
            P_f = (w + x) / N
            P_f_ = 1 - P_f

            P_cGf = w / (w + x)
            P_c_Gf = x / (w + x)

            WET1 = 0
            WET2 = 0

            if P_cGf != 1 and P_cGf != 0:
                WET1 = P_c * P_f * abs(np.log2( (P_cGf * (1 - P_c)) / (P_c * (1 - P_cGf)) ))
            if P_c_Gf != 1 and P_c_Gf != 0:
                WET2 = P_c_ * P_f * abs(np.log2( (P_c_Gf * (1 - P_c_)) / (P_c_ * (1 - P_c_Gf)) ))

            WET_ = WET1 + WET2

            WET.append(WET_)
        return WET

    def EFS(self, X, Y):
        Xi = []
        Xi.append(chi2(X, Y)[0])
        Xi.append(self.InformationGain(X, Y))
        Xi.append(self.MutualInformation(X, Y))
        Xi.append(self.CrossEntropy(X, Y))
        Xi.append(self.WeightOfEvidenceForText(X, Y))
        #Xi.append(mutual_info_classif(X, Y, discrete_features=True))

        t = len(Xi)                     # number of feature scoring algorithms
        if X.shape[1] > 500:
            w = int(X.shape[1]/100)     # window size
            tau_i = int(X.shape[1]/20)  # tau
            step_size = int(w / 5)      # step size for cross validation (ideally 1, but VERY slow performance)
        else:
            w = int(X.shape[1]/10)
            tau_i = int(X.shape[1]/2)
            step_size = 1

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
            cv_scores = cross_val_score(MultinomialNB(), candidate_features, Y, cv=5, scoring='accuracy')
            scores.append(cv_scores.mean())
            print("%d - Cross Validation Accuracy: %0.4f (+/- %0.2f)" % (i, cv_scores.mean(), cv_scores.std()))

        best_score_index = scores.index(max(scores))
        best_score = scores[best_score_index]

        print("Best Scoring Index: %d" % (best_score_index))

        candidate_feature_indexes = OptCandFeatures[best_score_index]

        return candidate_feature_indexes