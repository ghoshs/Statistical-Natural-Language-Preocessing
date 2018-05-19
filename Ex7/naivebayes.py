import os
from nltk import RegexpTokenizer
from collections import Counter
import math

# create a unigram dictionary from tokens along with their absolute frequencies
def create_unigram_dict(tokens):
	unigram_dict = dict(Counter(tokens))
	return unigram_dict

# return the tokenized text
def tokenize(text):
	# create a word tokenizer
	tokenizer = RegexpTokenizer(r'\w+')
	# tokenize the text into words 
	tokens = tokenizer.tokenize(text)
	# convert text to lower case and ignore all unconvertable utf8 characters since these are likely punctuation marks
	tokens = [x.lower().encode('ascii','ignore') for x in tokens]
	return tokens

def unigram_distribution_test(path, filename):
	text = open(path+'/'+filename, 'r').read().decode('utf8')
	tokens = tokenize(text)
	return create_unigram_dict(tokens), len(tokens)

def unigram_distribution(path, classlist):
	text = ''
	for document in classlist:
		text += open(path+'/'+document, 'r').read().decode('utf8')
		text += '\n'
	tokens = tokenize(text)
	return create_unigram_dict(tokens), len(tokens)

def classification(class1_path, class2_path, test_path):

	# ----------------- TRAINING -------------------

	class1 = [x for x in os.listdir(class1_path) if x.endswith('.txt')]
	class2 = [x for x in os.listdir(class2_path) if x.endswith('.txt')]

	K = len(class1) + len(class2)
	# class frequency is represented by document count of each author
	Pc1 = len(class1) / float(K)
	Pc2 = len(class2) / float(K)

	unigram_c1, Nc1 = unigram_distribution(class1_path, class1)
	unigram_c2, Nc2 = unigram_distribution(class2_path, class2)

	# features is a vocabulary of all unique terms in training data with a count of occurences of each feature in all documents of a particular class
	# features[f] = [# occurence of f over all docs in c1, # occurences of f over all docs in c2]
	features = {i:[unigram_c1[i], 0] for i in unigram_c1} 

	for i in unigram_c2:
		if i not in features:
			features[i] = [0, unigram_c2[i]]
		else:
			features[i][1] = unigram_c2[i]

	V = len(features)
	# tried with both values, got the same result.
	# alpha = 0.3 
	alpha = 0.9*V / float(Nc1+Nc2)

	P_f_given_c = {} # for each feature f in features, P_f_given_c[f] = [P(f|c1), P(f|c2)] using Lidstone smoothing

	for f in features:
		P_f_given_c[f] = []
		P_f_given_c[f].append((alpha + features[f][0]) / float(alpha*V + Nc1)) # P(f|c1) = (alpha + #occ of f in c1)/(#tokens in c1 + alpha*V)
		P_f_given_c[f].append((alpha + features[f][1]) / float(alpha*V + Nc2)) # P(f|c2)

	reserved_c1 = alpha / float(alpha*V + Nc1)
	reserved_c2 = alpha / float(alpha*V + Nc2)

	print('reserved_c1: '+str(Nc1)+' reserved_c2: '+str(Nc2))

	# ----------------- PREDICTION -------------------

	test = [x for x in os.listdir(test_path) if x.endswith('.txt')]
	# take log values since score is a product of probabilities
	for document in test:
		unigram_test, Nt = unigram_distribution_test(test_path, document)
		score_c1 = math.log(Pc1)
		score_c2 = math.log(Pc2)
		for w in unigram_test:
			score_c1 += math.log(P_f_given_c[w][0]) if w in features else math.log(reserved_c1)
			score_c2 += math.log(P_f_given_c[w][1]) if w in features else math.log(reserved_c2)
		C = "author1" if score_c1 > score_c2 else "author2"
		print('Document: ' + document + ' Predicted author: ' + C + ' Sc1: ' + str(score_c1) + ' Sc2: ' + str(score_c2))

classification('Materials_Ex7/author1', 'Materials_Ex7/author2', 'Materials_Ex7/test_author')