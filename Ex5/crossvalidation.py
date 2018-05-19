import nltk
from nltk.tokenize import RegexpTokenizer
import math
from collections import Counter
from itertools import izip
import matplotlib.pyplot as plt

def perplexity_bigram(test_unigram_freq, test_bigram_freq, train_unigram_freq, train_cond_prob, alpha, V, N, reserved, alphah, Ntrain):
	pp = 0.0
	for wh in test_bigram_freq: # for every unique sequence in test data

		pw_given_h = 0.0
		pw = train_unigram_freq[wh[1]]/float(Ntrain) if wh[1] in train_unigram_freq else reserved

		# if the history h of the bigram (wh) is not found in the training corpus, alphah = 1.0 such that P(w|h) is the estimate of P(w) 
		ah = alphah[wh[0]] if wh[0] in alphah else 1.0 
		
		# check if bigram is present in train data, ie, N(w,h) > 0
		if wh not in train_cond_prob: # N(w,h) == 0
			pw_given_h = ah * pw
		else:						  # N(w,h) > 0
			pw_given_h = train_cond_prob[wh]
		pp = pp + (test_bigram_freq[wh] / float(N)) * math.log(pw_given_h)
	pp = math.exp(-pp)
	return pp

# returns the smoothed bigram probability distribution P(w|h)
def smooth_bigram(Nwh, d, Pw, Nh, R, alphah):
	Pwh = {}

	for wh in Nwh:
		pw = Pw[wh[1]]  # wh[0] = history word; wh[1] = current word; Pw = P(w)
		# alphah = a(h) = dR(h)/N(h)
		Pwh[wh] = (Nwh[wh] - d) / float(Nh[wh[0]]) + alphah[wh[0]] * pw

	return Pwh 

# returns the smoothed unigram distribution and the reserved probability for unseen words
def smooth_unigram(alpha, N, d, Nw):
	smooth_p = {}
	V = len(Nw)
	for w in Nw:
		smooth_p[w] = (Nw[w] -d) / float(N) + alpha /float(V)

	return smooth_p, alpha/float(V)

# returns the backing-off weight
def get_alphah(R, d, N):
	alphah = {}
	for h in N:
		alphah[h] =  d * R[h] / float(N[h])
	return alphah

# return R for all words such that R(h) = # bigrams such that h is the 1st word of the bigram
def create_R_unigram(unigram_freq, bigram_freq):
	R = {}
	for h in unigram_freq:
		for wh in bigram_freq:
			if h is wh[0]:  # N(wh)>0
				if h not in R:
					R[h] = 1
				else:
					R[h] = R[h] + 1
		# when the word is the last word in the corpus and occurs only once, it is not the history for any other word in the vocabulary
		if h not in R:
			R[h] = 0 
	return R

# create a bigram dictionary from tokens along with their absolute frequencies
def create_bigram_dict(tokens):
	bigram_dict = dict(Counter(izip(tokens, tokens[1:])))
	return bigram_dict

# create a unigram dictionary from tokens along with their absolute frequencies
def create_unigram_dict(tokens):
	unigram_dict = dict(Counter(tokens))
	return unigram_dict

# return the tokenized text
def tokenize(filename):
	text = open(filename, 'r').read().decode('utf8')
	# tokens = text.split()
	# return tokens
	# create a word tokenizer
	tokenizer = RegexpTokenizer(r'\w+')
	# tokenize the text into words 
	tokens = tokenizer.tokenize(text)
	# convert text to lower case and ignore all unconvertable utf8 characters since these are likely punctuation marks
	tokens = [x.lower().encode('ascii','ignore') for x in tokens]
	return tokens

# perform k cross validation
def k_cross_validate(d, k, filename):
	tokens = tokenize(filename)
	l = len(tokens)
	pp = 0.0
	for i in range(k):
		# print('Fold = '+str(i+1))
		start_idx = (l/k) * i
		end_idx = start_idx + (l/k) if i < k-1 else l

		test = tokens[start_idx: end_idx]
		train = tokens[0:start_idx]
		for i in range(l - end_idx):
			train.append(tokens[end_idx + i])

		# total tokens in test and train corpus
		N_test = len(test)
		N_train = len(train)

		unigram_freq_train = create_unigram_dict(train)
		bigram_freq_train = create_bigram_dict(train)

		V = len(unigram_freq_train)
		alpha = d*V*V / float(N_train)

		# return P(w): smoothed unigram distribution
		unigram_smooth, uni_reserved_prob = smooth_unigram(alpha, N_train, d, unigram_freq_train)

		# create R(h) for all unigrams
		R = create_R_unigram(unigram_freq_train, bigram_freq_train)

		alphah = get_alphah(R, d, unigram_freq_train)
		# return P(w|h) for all train bigrams 
		bigram_smoothed_cond_train = smooth_bigram(bigram_freq_train, d, unigram_smooth, unigram_freq_train, R, alphah)

		unigram_freq_test = create_unigram_dict(test)
		bigram_freq_test = create_bigram_dict(test)

		pp = pp + perplexity_bigram(unigram_freq_test, bigram_freq_test, unigram_freq_train, bigram_smoothed_cond_train, alpha, V, N_test, uni_reserved_prob, alphah, N_train)

	return pp/float(k)

def cross_validation(filename):
	d = [x/10.0 for x in list(range(1, 11))]
	perplexity = []
	k = 5

	for i in range(len(d)):
		print('Evaluating for d='+str(d[i]))
		perplexity.append(k_cross_validate(d[i], k, filename))
		print('Perplexity = '+str(perplexity[len(perplexity)-1]))

	d_pt = d[perplexity.index(min(perplexity))]
	print('Optimal value for discounting parameter, d = '+str(d_pt))

	bi, = plt.plot(d, perplexity, '-bo', label = "perplexity", linewidth = 2.0)
	plt.axhline(y=min(perplexity), color='g', linestyle='-')
	plt.xlabel('Discounting parameter (d)')
	plt.ylabel('Cross-Validation Perplexity')
	plt.legend(handles = [bi], numpoints = 1)
	plt.grid(True)
	plt.savefig('cross_validation_perplexity.png')
	plt.show()

cross_validation('./materials_ex5/text.txt')