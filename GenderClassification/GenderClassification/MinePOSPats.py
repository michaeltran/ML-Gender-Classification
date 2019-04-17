import numpy as np

MAX_LENGTH = 7

class MinePOSPats(object):
    CountDict = {}
    D = []
    T = []
    minSupport = 0
    minAdherence = 0

    def __init__(self, D, minSupport, minAdherence):
        self.D = D
        self.T = self.GetAllPossiblePOS(self.D)
        self.minSupport = int(len(self.D) * minSupport)
        self.minAdherence = minAdherence

    #def ConvertToPOS(self, D_words):
    #    D = []
    #    T = []
    #    for d_words in D_words:
    #        d = []

    #        sentence = Sentence(d_words, use_tokenizer=True)
    #        TAGGER.predict(sentence)
    #        for token in sentence:
    #            pos = token.get_tag('pos').value
    #            d.append(pos)
    #            if pos not in T:
    #                T.append(pos)

    #        #tokens = nltk.word_tokenize(d_words)
    #        #tagged = nltk.pos_tag(tokens)
    #        #for (word, pos) in tagged:
    #        #    d.append(pos)
    #        #    if pos not in T:
    #        #        T.append(pos)
    #        D.append(d)
    #        print(len(D))
    #    return D, T

    def GetAllPossiblePOS(self, D):
        T = []
        for d in D:
            for tag in d:
                if tag not in T:
                    T.append(tag)
        return T

    def MinePOSPats(self):
        print('Start POS Mining')
        print('Total Amount of Documents: %d' % (len(self.D)))
        print('Total Amount of Unique POS: %d' % (len(self.T)))
        print('Minimum Support: %d' % (self.minSupport))
        print('Minimum Adherence: %0.2f' % (self.minAdherence))
        C = []
        F = []
        SP = []

        C_k = []
        F_k = []
        for t in self.T:
            C_k.append(t)
        self.GetkgramCounts(1)
        for c in C_k:
            if self.GetCount([c]) >= self.minSupport:
                F_k.append([c])

        C.append(C_k)
        F.append(F_k)
        SP.append(F_k)

        for k in range(2, MAX_LENGTH + 1):
            C_k = self.CandidateGen(F[k-1-1])
            F_k = []
            SP_k = []

            if len(C_k) > 0:
                self.GetkgramCounts(k)
                for c in C_k:
                    if self.GetCount(c) >= self.minSupport:
                        F_k.append(c)
                for f in F_k:
                    if self.FairSCP(f) >= self.minAdherence:
                        SP_k.append(f)
            else:
                print('C_k empty at = %d' % (k))
            C.append(C_k)
            F.append(F_k)
            SP.append(SP_k)

            if len(SP_k) == 0:
                print('F_k length = %d' % (len(F_k)))
                print('SP_k empty at = %d' % (k))
                break

        print('Stopped at k = %d' % (k))
        result = []
        for SP_k in SP:
            result += SP_k
        print('Extracted POS Patterns: %d' % (len(result)))
        return result

    def CandidateGen(self, F_k_1):
        C_k = []
        for c in F_k_1:
            for t in self.T:
                c_prime = c + [t]
                C_k.append(c_prime)

        return C_k

    def FairSCP(self, f):
        n = len(f)

        top = np.power(self.GetCount(f) / len(self.D), 2)
        bottom = 0
        for i in range(1, n):
            features = np.split(f, [i])
            features = np.array_split(f, [i])
            pr_feature_1 = self.GetCountSpecial(features[0].tolist()) / len(self.D)
            pr_feature_2 = self.GetCountSpecial(features[1].tolist()) / len(self.D)
            bottom += (pr_feature_1 * pr_feature_2)
        bottom = (1 / (n - 1)) * bottom
            
        result = top / bottom
        return result

    def GetCount(self, f):
        f_key = ' '.join(f)

        if f_key in self.CountDict:
            return self.CountDict[f_key]
        else:
           return 0

    def GetCountSpecial(self, f):
        f_key = ' '.join(f)

        if f_key in self.CountDict:
            return self.CountDict[f_key]
        else:
            print('Special Scenario Hit')
            # SPECIAL SCENARIO ONLY - SHOULD NEVER HAPPEN
            count = 0
            for d in self.D:
                for i in range(len(d) - len(f) + 1):
                    x = []
                    for j in range(len(f)):
                        x.append(d[i + j])
                    if x == f:
                        # Only need to count once per document
                        count += 1 
                        break
            self.CountDict[f_key] = count
            return count

    def GetkgramCounts(self, k):
        tempDict = {}
        for d in self.D:
            ngrams = []
            for i in range(len(d) - k + 1):
                x = []
                for j in range(k):
                    x.append(d[i + j])
                if x not in ngrams:
                    ngrams.append(x)
            for ngram in ngrams:
                f_key = ' '.join(ngram)
                if f_key in tempDict:
                    tempDict[f_key] += 1
                else:
                    tempDict[f_key] = 1
        for key in tempDict:
            self.CountDict[key] = tempDict[key]