import os
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import math
import operator

ps = PorterStemmer()
wl = WordNetLemmatizer()

# create a unigram dictionary from tokens along with their absolute frequencies
def create_unigram_dict(tokens):
	unigram_dict = dict(Counter(tokens))
	return unigram_dict

def tokenize(text):
	# create a word tokenizer
	tokenizer = RegexpTokenizer(r'\w+')
	# tokenize the text into words 
	tokens = tokenizer.tokenize(text)
	# convert text to lower case and ignore all unconvertable utf8 characters since these are likely punctuation marks
	tokens = [x.lower().encode('ascii','ignore') for x in tokens]
	return tokens

def preprocess(doc, pathname, stopfile):
	text = open(pathname+'/'+doc, 'r').read().decode('utf8')
	tokens = tokenize(text)
	stopwords = open(stopfile, 'r').read().decode('utf8').split('\n')

	stop_removed = [x for x in tokens if x not in stopwords]

	# nltk WordNetLemmatizer does not reduce verb inflections to root unless, an argument pos='v' is explicitly mentioned. 
	# Moreover, in nltk Stem(lemmatize(text)) = Lemmatize(Stem(text)) = Stem(text) if we are not concered about the root word exprssed 
	# in its correct morphological form. 
	# Thus, we use only Porter's Stemming Algorithm.
	stemmed = [ps.stem(i).encode('ascii','ignore') for i in stop_removed]
	# print(stemmed[0:10])
	return stemmed

def feature_selection(pathname, stopfile):

	tokens_list = [] # list of all tokens in each class represented by doclist
	unigram_list = [] # list of unigram frequency in each class
	classlist = [x for x in os.listdir(pathname) if x.endswith('.txt')]
	N = 0 # total number of tokens in the training corpus
	# assuming class probablity is represented by document frequency which in this case is equal
	P_c = (1/float(len(classlist)))
	V = {}

	# tokenize all documents in each class and extend the vocabulary
	for doc in classlist: 
		tokens = preprocess(doc, pathname, stopfile)
		unigram = create_unigram_dict(tokens)
		tokens_list.append(tokens)
		unigram_list.append(unigram)
		for t in unigram:
			if t not in V:
				V[t] = unigram[t]
			else:
				V[t] += unigram[t]
		N += len(tokens)

	# PMI of (token, class) for each class
	PMI = {}
	# tried with both values of alpha got the same result for top 10 features.
	# alpha = 0.3
	alpha = 0.9*len(V) / float(N)
	for t in V:
		pmi = None
		for i in range(len(classlist)):
			Ni = len(tokens_list[i])
			P_tc = (unigram_list[i][t] + alpha)/ float(Ni + alpha*len(V)) if t in unigram_list[i] else alpha / float(Ni + alpha*len(V))
			P_t = V[t] / float(N)
			curr_pmi = math.log(P_tc / float(P_t * P_c))
			if pmi == None:
				pmi = curr_pmi
			else:
				pmi = curr_pmi if curr_pmi > pmi else pmi
		PMI[t] = pmi

	idx = max(PMI, key = PMI.get)
	top_ten = sorted(PMI.items(), key=operator.itemgetter(1), reverse=True)[0:10]

	print('--------------------------------------------------------------------------')
	print('Original Feature Size: ' + str(len(V)))
	print('--------------------------------------------------------------------------')
	print('Top 10 features and their PMI')
	for i in top_ten:
		print i

		# pmi_list.append(pmi)



feature_selection('./Materials_Ex7/train', './Materials_Ex7/stopwords.txt')