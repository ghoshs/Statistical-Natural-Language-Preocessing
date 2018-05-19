import numpy
import nltk
import math
import numpy as np
import matplotlib.pyplot as plt

# read and preprocess sample text
def preprocess(filename):
	fp = open(filename)
	text = fp.read().decode("utf-8-sig")
	fp.close()

	tokens = nltk.word_tokenize(text)
	tokens = [x.lower().encode('ascii','ignore') for x in tokens]

	punctuations = ['?','!', '(', ')', ',', '.', '_', '-', ':', ';']

	# remove all punctuation tokens
	for item in punctuations:
		tokens = [x for x in tokens if x != item]

	# remove all punctuations
	new_tokens = []
	for idx, token in enumerate(tokens):
		modified = ''.join([char for char in token if char not in punctuations])
		new_tokens.append(modified)
	# remove all empty tokens
	new_tokens = [x for x in new_tokens if x != '']	
	return new_tokens

def create_unigram_dict(tokens):
	unigram_dict = {}

	for token in tokens:
		if token not in unigram_dict:
			unigram_dict[token] = 1
		else:
			unigram_dict[token] = unigram_dict[token] + 1
	return unigram_dict

def create_bigram_dict(tokens):
	bigram_dict = {}
	for idx in range(len(tokens) - 1):
		bigram = tokens[idx] + ' ' + tokens[idx + 1] 
		if bigram not in bigram_dict:
			bigram_dict[bigram] = 1
		else:
			bigram_dict[bigram] = bigram_dict[bigram] + 1
	return bigram_dict

def perplexity_unigram(test_word_freq, train_word_prob, alpha, v, test_tokens):
	pp = 0.0
	for word in test_word_freq:
		pw = 0.0
		if word not in train_word_prob:
			pw = alpha / float(alpha * float(v))
		else:
			pw = (train_word_prob[word] + alpha) / float(alpha * float(v))
		pp = pp + (test_word_freq[word] / float(test_tokens)) * math.log(pw)
	pp = math.exp(-pp)
	return pp

def perplexity_bigram(test_sequence_freq, train_sequence_prob, train_conditional_prob, alpha, v, test_tokens):
	pp = 0.0
	for sequence in test_sequence_freq: # for every unique sequence
		pw_given_h = 0.0
		# check if sequence is present in train data
		if sequence not in train_sequence_prob:
			pw_given_h = alpha / float(alpha * float(v))
		else:
			pw_given_h = (train_sequence_prob[sequence] + alpha) / float(train_conditional_prob[sequence.split(' ')[0]] + alpha * float(v))
		pp = pp + (test_sequence_freq[sequence] / float(test_tokens)) * math.log(pw_given_h)
	pp = math.exp(-pp)
	return pp


# call preproccessing for train sample
tokens = preprocess('Materials/English1.txt')
total_tokens = len(tokens)

# create unigram dictionary
unigram_dict = create_unigram_dict(tokens)
bigram_dict = create_bigram_dict(tokens)

bigram_relative_freq = {token: bigram_dict[token] / float(total_tokens) for token in bigram_dict}


# // smooth over conditional in order to reserve probability for unknown words... p(<unknown|history> = ...)
unigram_conditional = {token: unigram_dict[token] / float(total_tokens) for token in unigram_dict}

# preprocecss test sample
tokens_test = preprocess('Materials/English2.txt')

test_word_freq = create_unigram_dict(tokens_test)
test_sequence_freq = create_bigram_dict(tokens_test)

alpha = 0.3

#calculate perplexity

pp_unigram = perplexity_unigram(test_word_freq, unigram_conditional, alpha, len(unigram_dict), len(tokens_test))
pp_bigram = perplexity_bigram(test_sequence_freq, bigram_relative_freq, unigram_conditional, alpha, len(unigram_dict), len(tokens_test))

print('*******************************************************')
print('Total word choices:' + str(len(unigram_dict)))
print('Perplexity Unigram: ' + str(pp_unigram))
print('Perplexity Bigram: ' + str(pp_bigram))
print('*******************************************************')

pp_bi = []
pp_max = []
for i in xrange(20, 101, 20):
	# create fractional corpus
	corpus_length = int((i / float(100)) * len(tokens))
	frac_token = tokens[0: corpus_length]
	total_tokens = len(frac_token)

	unigram_dict = {}
	bigram_dict = {}
	bigram_relative_freq = {}
	unigram_conditional = {}
	# create dictionaries
	unigram_dict = create_unigram_dict(frac_token)
	bigram_dict = create_bigram_dict(frac_token)

	bigram_relative_freq = {token: bigram_dict[token] / float(total_tokens) for token in bigram_dict}
	unigram_conditional = {token: unigram_dict[token] / float(total_tokens) for token in unigram_dict}

	pp_bigram = perplexity_bigram(test_sequence_freq, bigram_relative_freq, unigram_conditional, alpha, len(unigram_dict), len(tokens_test))
	pp_max.append(len(unigram_dict))
	pp_bi.append(pp_bigram)

print('Percentage of text corpus used: 20, 40, 60, 80, 100')
print('Bigram perplexity: ', pp_bi)
print('Max perplexity: ', pp_max)
# # plot bigram perplexity graph
# bi, = plt.plot(np.arange(20, 101, 20), pp_bi, '-bo', label = "bigram", linewidth = 2.0)
# plt.xlabel('%age of text corpus used ')
# plt.ylabel('Perplexity')
# plt.legend(handles = [bi], numpoints = 1)
# plt.savefig('perplexity_bigram.png')