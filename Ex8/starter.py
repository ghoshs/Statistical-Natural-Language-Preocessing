from nltk.corpus import senseval
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

import operator
import math
import numpy as np

hard_f, interest_f, line_f, serve_f = senseval.fileids()

class sample(object):
    def __init__(self, inst):
        self.label=inst.senses[0]

        p = inst.position
        context = [tuple[0]  for tuple in inst.context[p-5:p] if len(tuple)>1] #checking if list element is actually a tuple, phrasal elements are not
        context+= [tuple[0] for tuple in inst.context[p+1:p+6] if len(tuple)>1]
        self.context=context 
		
def tokenize(text):
    # lowercase
    text = text.lower()
    # create a word (alphanumeric and underscore) tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # tokenize the text into words 
    tokens = tokenizer.tokenize(text)
    # remove stopwords
    tokens = [x for x in tokens if x not in stopwords.words('english')]
    return tokens

def get_P_xf_t(P, Q, samples):
    N = 0 # count all combinations of (xf, t)
    P_xf_t = {t:{xf: 0 for xf,_ in Q} for t in P}
    for sample in samples:
        t = sample.label
        for xf in sample.context:
            if xf in P_xf_t[t]:
                P_xf_t[t][xf] += 1 # count of (xf, t)
                N += 1
    # normalize
    P_xf_t = {t:{xf: P_xf_t[t][xf] / float(N) for xf in P_xf_t[t]} for t in P_xf_t}
    return P_xf_t

def flipflop(word):
    # instance for word 
    instances = senseval.instances(word)
    samples = []
    P = {}
    PMImax = {}
    indicator = {}
    corpus = '' # corpus comprising the 10 context words of all occurences of the ambiguous word
    N = 0 # total length of corpus = #occurences of ambiguous words + #context words in each occurence
    for inst in instances:
        s = sample(inst)
        # all training samples as a list
        samples.append(s)
        # create P a list of senses
        if s.label not in P:
            P[s.label] = 1
        else:
            P[s.label] += 1
        corpus += (' ').join(s.context)
        N += 1 + len(s.context) # #context words maybe less than 10

    # print('Senses: ', P)

    tokens = tokenize(corpus)
    v = dict(Counter(tokens))
    # get 10 most frequent context words for the current ambiguous word
    Q = sorted(v.items(), key=operator.itemgetter(1), reverse=True)[0:10]
    P_xf_t = get_P_xf_t(P, Q, samples)
    pmi = {t:{xf: 0 for xf,_ in Q} for t in P}
    for t in P:
        indicator[t] = []
        for xf, f in Q:
            p_xf = f / float(N)
            p_t = P[t] / float(N)
            if P_xf_t[t][xf] == 0.0:
                pmi[t][xf] = 0.0 # if P(xf,t) == 0 => xf and t are statistically independent and P(xf,t) = P(xf) * P(t)
            else:
                # print(pmi[t][xf], P_xf_t[t][xf], p_xf, p_t)
                pmi[t][xf] = P_xf_t[t][xf] * math.log(P_xf_t[t][xf]/float(p_xf * p_t))
            if (t not in PMImax) or (pmi[t][xf] > PMImax[t]):
                PMImax[t] = pmi[t][xf]
                # indicator[t] = xf
        sort_pmi = sorted(pmi[t].items(), key=operator.itemgetter(1), reverse=True)
        indicator[t] = sort_pmi[0][0]
        # indicator[t] = [ind for ind, p in sort_pmi if p > 0.0]

    for s in indicator:
        print '{0:<10} : {1:{fill}{width}} : {2}'.format(s,PMImax[s],indicator[s], fill=' ', width=15)
    return indicator

if __name__=="__main__":
	
    indicator = {}
    sense = {}
    print '{0:<10} : {1:<10} : {2:<10}'.format('Sense','PMImax','Indicator')
    print '------------------------------------'
    print 'Word: hard'
    flipflop(hard_f)
    print '------------------------------------'
    print 'Word: interest'
    flipflop(interest_f)
    print '------------------------------------'
    print 'Word: line'
    flipflop(line_f)
    print '------------------------------------'
    print 'Word: serve'
    flipflop(serve_f)
    # for word in indicator:
    #     print(word)
    #     for s in indicator[word]:
    #         print(s+' : '+indicator[word][s])