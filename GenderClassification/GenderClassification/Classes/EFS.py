import numpy as np
import random
from scipy.sparse import issparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score

from Classes.FSC import FSC
from Helper.DebugPrint import DebugPrint

class EFS(object):
    def GetContingencyTable(self, X, Y, feature):
        #   1 0
        # 1 w x
        # 0 y z

        # X_When_Y_1 = Values of X when Y = 1
        # temp = indexes when feature > 0

        if feature < len(self.ContingencyTableDict):
            w, x, y, z = self.ContingencyTableDict[feature]
        else:
            # Full Array
            feature_column = X[:,feature]
            X_When_Y_1 = np.extract(Y, feature_column)
            temp = np.where(X_When_Y_1 > 0)
            w = float(temp[0].size)

            temp = np.where(feature_column > 0)
            x = float(temp[0].size - w)

            temp = np.where(X_When_Y_1 == 0)
            y = float(temp[0].size)

            temp = np.where(feature_column == 0)
            z = float(temp[0].size - y)

            ## Sparse Array
            #indices = np.where(Y)[0]
            #feature_column = X[:,feature]
            #X_When_Y_1 = feature_column[indices,:]

            #w = float(X_When_Y_1.nnz)
            #x = float(feature_column.nnz - w)
            #y = float(X_When_Y_1.shape[0] - w)
            #z = float(feature_column.shape[0] - feature_column.nnz - y)

            self.ContingencyTableDict.append((w, x, y, z))

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

        #pool = mp.Pool(mp.cpu_count())

        #results = [pool.apply(CalculateChi2, args=[i, X, Y, N]) for i in range(X.shape[1])]
        #pool.close()
        #print(results)

        #pool = mp.Pool(mp.cpu_count())
        #def collect_result(result):
        #    global results
        #    results.append(result)

        #for i in range(X.shape[1]):
        #    pool.apply_async(CalculateChi2, args=[i, X, Y, N], callback=collect_result)

        #pool.close()
        #pool.join()

        #results.sort(key=lambda x: x[0])
        #results_final = [r for i, r in results]

        for i in range(X.shape[1]):
            w, x, y, z = self.GetContingencyTable(X, Y, i)

            if (w + y) == 0 or (x + z) == 0 or (w + x) == 0 or (y + z) == 0:
                chi2_fc = 0
            else:
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

            P_cGf = 0
            P_c_Gf = 0

            if w + x != 0:
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

            P_cGf = 0
            P_c_Gf = 0

            if w + x != 0:
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

    def RandomSampleX(self, X, Y):
        indices = []

        indices = random.sample(range(0, X.shape[0]), min(X.shape[0], 1000))
        new_X = X[indices,:]
        new_Y = [Y[i] for i in indices]

        return new_X, new_Y

    def EFS(self, X, Y, classifier, feature_selections):
        self.ContingencyTableDict = []

        # Create Y_mask (Convert -1 to 0)
        Y_mask = np.array(Y)
        Y_mask[Y_mask < 0] = 0
        Y_mask = Y_mask.tolist()

        if issparse(X):
            X_dense, Y_mask = self.RandomSampleX(X, Y_mask)
            X_dense = X_dense.toarray()
        else:
            X_dense = X

        Xi = []
        for feature_selection in feature_selections:
            if FSC.CHI == feature_selection:
                #Xi.append(chi2(X, Y)[0])
                Xi.append(self.ChiSquared(X_dense, Y_mask))
            elif FSC.IG == feature_selection:
                Xi.append(self.InformationGain(X_dense, Y_mask))
            elif FSC.MI == feature_selection:
                Xi.append(self.MutualInformation(X_dense, Y_mask))
            elif FSC.CE == feature_selection:
                Xi.append(self.CrossEntropy(X_dense, Y_mask))
            elif FSC.WOE == feature_selection:
                Xi.append(self.WeightOfEvidenceForText(X_dense, Y_mask))

        #Xi.append(self.ChiSquared(X_dense, Y))
        #Xi.append(mutual_info_classif(X, Y, discrete_features=True))

        del X_dense
        del Y_mask
        del self.ContingencyTableDict

        t = len(Xi)                     # number of feature scoring algorithms
        if X.shape[1] > 500:
            #w = int(X.shape[1]/100)     # window size
            #tau_i = int(X.shape[1]/20)  # tau
            #step_size = int(w / 5)      # step size for cross validation (ideally 1, but VERY slow performance)

            w = int(X.shape[1]/2) - 1
            tau_i = int(X.shape[1]/2)
            step_size = int(w / 15)
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
                    #zeta_i = np.argsort(Xi[i])[:tau]
                    C_i.append(zeta_i)
                iterator += 1
            C.append(C_i)

        del Xi

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
            cv_scores = cross_val_score(classifier, candidate_features, Y, cv=10, scoring='accuracy', n_jobs=5)
            scores.append(cv_scores.mean())
            DebugPrint("%d - Cross Validation Accuracy: %0.4f (+/- %0.2f)" % (i, cv_scores.mean(), cv_scores.std()))
            print("%d - Cross Validation Accuracy: %0.4f (+/- %0.2f)" % (i, cv_scores.mean(), cv_scores.std()))

        best_score_index = scores.index(max(scores))
        best_score = scores[best_score_index]

        DebugPrint("Best Scoring Index: %d" % (best_score_index))
        DebugPrint("Best Score: %0.4f" % (best_score))
        DebugPrint("Best Scoring Index: %d, Best Score: %0.4f" % (best_score_index, best_score))

        candidate_feature_indexes = OptCandFeatures[best_score_index]

        return candidate_feature_indexes