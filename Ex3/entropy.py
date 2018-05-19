import math
import nltk
from nltk import word_tokenize
from collections import Counter
from itertools import izip
import operator

# 1. vocabulary  = Vp KL(p||q) . for x in Vp if x is not in Vq, q(x) = alpha /(N + alpha * Vq) OR
# 2. smooth approximating function q

# compute the KL Divergence between two texts where filename two forms the approximating distribution
def KL_divergence(P, Q):

	DLk = 0.0
	alpha = 0.1

	# preprocess corpus to return tokens
	tokens_P = preprocess(P)
	tokens_Q = preprocess(Q)

	N_P = len(tokens_P)
	N_Q = len(tokens_Q)

	unigram_P = create_unigram_dict(tokens_P)
	unigram_Q = create_unigram_dict(tokens_Q)

	vocab = [x for x in unigram_P]

	# calculate the smoothed probabilities of Q - Q is an approximator of P
	smoothed_Q = lidstone_probability(unigram_Q, vocab, alpha, N_Q)

	# KL Diververgence where probability distribution of Q is the approximator of distribution of P 
	for i in vocab:
		pi = unigram_P[i]/float(N_P)
		qi = smoothed_Q[i]
		DLk += pi*math.log(pi/qi)

	return DLk

# lower case, remove stop words and tokenize a given text
def preprocess(filename):
	fp = open(filename)
	text = fp.read().decode("utf-8")
	fp.close()

	tokens = word_tokenize(text)
	# remove epilogue and prologue regarding Project Gutenberg in the text
	idx1 = tokens.index('***')
	idx2 = idx1 + 1 + tokens[idx1 + 1: len(tokens)].index('***') # locate the position of *** just before start of the text
	idx3 = idx2 + 1 + tokens[idx2 + 1: len(tokens)].index('***') # locate the position of *** just before end of the text
	tokens = tokens[idx2 : idx3] # remove all unrelated text
	# convert text to lower case and ignore all unconvertable utf8 characters since these are likely punctuation marks
	tokens = [x.lower().encode('ascii','ignore') for x in tokens]
	# create a list of punctuations supported by ascii which may still be present in the text
	punctuations = ['?','!', '(', ')', ',', '.', '_', '-', ':', ';', "'"]

	# remove all punctuation tokens
	for item in punctuations:
		tokens = [x for x in tokens if x != item]

	# remove all punctuations occuring with string characters
	new_tokens = []
	for idx, token in enumerate(tokens):
		modified = ''.join([char for char in token if char not in punctuations])
		new_tokens.append(modified)
	# remove all empty tokens
	new_tokens = [x for x in new_tokens if x != '']	
	return new_tokens

# create a dictionary from token along with their absolute frequencies
def create_unigram_dict(tokens):
	unigram_dict = dict(Counter(tokens))
	return unigram_dict

# calculate the lidstone probability of a distribution over the given vocabulary to account for unknown words present in the vocab
def lidstone_probability(unigram_freq, vocab, alpha, N):
	v = len(vocab)
	smoothed = {}
	for word in vocab:
		if word in unigram_freq:
			smoothed[word] = (unigram_freq[word] + alpha) / (N + float(alpha) * float(v))
		else:
			smoothed[word] = alpha / (N + float(alpha) * float(v))
	# print('___________________________'+str(max(smoothed.values()))+'__________________________'+str(min(smoothed.values())))
	return smoothed

def create_bigram_dict(tokens):
	bigram_dict = dict(Counter(izip(tokens, tokens[1:])))
	return bigram_dict

# calculate pointwise I(x,y) for all pairs of words in the file
def calculateIXY(filename):
	tokens_file = preprocess(filename)
	N_file = len(tokens_file)

	# create the frequency distributions
	unigram_text = create_unigram_dict(tokens_file)
	bigram_text = create_bigram_dict(tokens_file)

	Ixy = {}

	# calculate Ixy for each word pair (previous_word, next_word) = E(next_word) + E(next_word | previous_word)
	for pair in bigram_text:
		x = pair[0]
		y = pair[1]
		# calculate probability of next word
		p_y = unigram_text[y] / float(N_file)
		# calculate conditional prob of y (next_word) given x (previous_word)
		p_yx = bigram_text[pair] / float(unigram_text[x])
		# Ixy[pair] = (-p_y * math.log(p_y)) + (p_yx * math.log(p_yx)) WRONG FORMULA
		Ixy[pair] = -math.log(p_y) + math.log(p_yx)

	sortedI = sorted(Ixy.items(), key=operator.itemgetter(1))
	Imaxpair = sortedI[len(sortedI) -1][0]
	Imax = sortedI[len(sortedI) -1][1]
	Iminpair = sortedI[0][0]
	Imin = sortedI[0][1]
	return Imax, Imaxpair, Imin, Iminpair

def entropy(file1, file2, file3):
	print('----------------KL Divergence---------------------')
	kld1 = KL_divergence(file1, file2)
	print('KL Divergence between English1 and English2: ' + str(kld1))
	kld1 = KL_divergence(file2, file1)
	print('KL Divergence between English2 and English1: ' + str(kld1))
	kld2 = KL_divergence(file1, file3)
	print('KL Divergence between first English and German texts: '+str(kld2))
	kld3 = KL_divergence(file2, file3)
	print('KL Divergence between second English and German texts: '+str(kld3))
	kld4 = KL_divergence(file3, file1)
	print('KL Divergence between German and first English texts: '+str(kld4))
	kld5 = KL_divergence(file3, file2)
	print('KL Divergence between German and second English texts: '+str(kld5))

	print('----------------compute I(X|Y)--------------------')
	Imax, maxpair, Imin, minpair = calculateIXY(file1)
	print('I(X|Y) for English1.txt:')
	print 'Max Pair: %s I(x, y): %f' % (maxpair, Imax)
	print 'Min Pair: %s I(x, y): %f' % (minpair, Imin)
	Imax, maxpair, Imin, minpair = calculateIXY(file2)
	print('I(X|Y) for English2.txt:')
	print 'Max Pair: %s I(x, y): %f' % (maxpair, Imax)
	print 'Min Pair: %s I(x, y): %f' % (minpair, Imin)
	Imax, maxpair, Imin, minpair = calculateIXY(file3)
	print('I(X|Y) for German.txt:')
	print 'Max Pair: %s I(x, y): %f' % (maxpair, Imax)
	print 'Min Pair: %s I(x, y): %f' % (minpair, Imin)

entropy('English1.txt', 'English2.txt', 'German.txt')