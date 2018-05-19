import os
import nltk
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import reuters
from collections import Counter
import math
import operator
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

ps = PorterStemmer()

# create a unigram dictionary from tokens along with their absolute frequencies
def create_unigram_dict(tokens):
	unigram_dict = dict(Counter(tokens))
	return unigram_dict

def binary_multi_label(categories, dataset):
	mlb = MultiLabelBinarizer()
	labels = []
	for doc in dataset:
		c = [x for x in reuters.categories(doc) if x in categories] # only take those labels which are present in the categories list
		labels.append(set(c)) # prepare data for mlb input

	labels = mlb.fit_transform(labels)

	return labels

def create_subset(corpus):
	train_set = {}
	train_cat = set()
	test_set = {}
	test_cat = set()

	# print('Original: #Files: ' + str(len(reuters.fileids())) + ' #Categories: ' + str(len(reuters.categories())))
	# c = reuters.categories()
	for fileid in corpus:
		if 'training/' in fileid:
			train_set[fileid] = corpus[fileid]
			file_cat = reuters.categories(fileid)
			train_cat.update(file_cat) 
		elif 'test/' in fileid:
			test_set[fileid] = corpus[fileid]
			file_cat = reuters.categories(fileid)
			test_cat.update(file_cat)

	categories = reuters.categories()
	# keep only those categories which are present in both test and training
	categories_set = [x for x in categories if x in train_cat.intersection(test_cat)]

	# print('After preprocessing: #Train: ' + str(len(train_set)) + ' #Test: ' + str(len(test_set)) + ' #Cat: ' + str(len(categories_set)))

	# remove all training files which are single labelled and corresponding label is not present in test set
	train_set = {fileid: corpus[fileid] for fileid in train_set if not (len(reuters.categories(fileid)) == 1 and reuters.categories(fileid)[0] not in test_cat)}

	# remove all test files which are single labelled and corresponding label is not present in train set
	test_set = {fileid: corpus[fileid] for fileid in test_set if not(len(reuters.categories(fileid)) == 1 and reuters.categories(fileid)[0] not in train_cat)}

	# print('After subset: #Train: ' + str(len(train_set)) + ' #Test: ' + str(len(test_set)) + ' #Cat: ' + str(len(categories_set)))
	return categories_set, train_set, test_set

# preprocess train and test data and save them locally
def preprocess():

	fileids = reuters.fileids()
	corpus = {}
	for i in range(len(fileids)):
		tokens = reuters.words(fileids[i])
		# lowercase
		tokens = [x.lower() for x in tokens]

		# remove stopwords
		stopwords = nltk.corpus.stopwords.words('english')
		tokens = [x for x in tokens if x not in stopwords]

		# perform stemming
		tokens = [ps.stem(x).encode('ascii','ignore') for x in tokens]

		# remove words of length < 3
		tokens = [x for x in tokens if len(x) > 2]
		if len(tokens) > 0: # add the doc tokens in the corpus only if #tokens after preprocessing > 0
			corpus[fileids[i]] = tokens
	return corpus

def classification():
	corpus = preprocess()
	categories, train_set, test_set = create_subset(corpus)
	train_labels = binary_multi_label(categories, train_set)
	test_labels = binary_multi_label(categories, test_set)

	# create vocabulary from all training data
	V = {}
	N = 0 
	unigram_list = {}
	for fileid in train_set:
		tokens = train_set[fileid]
		unigram = create_unigram_dict(tokens)
		unigram_list[fileid] = unigram
		for t in unigram:
			if t not in V:
				V[t] = unigram[t]
			else:
				V[t] += unigram[t]
		N += len(tokens)

	alpha = 0.3
	# create tf-idf feature vector for each document in train set
	X = np.zeros((len(train_set), len(V)))
	print('creating train data vector...')
	D = len(train_set)
	idf = {}
	for j, t in enumerate(V):
		nt = 0
		# for fileid in train_set:
		# 	nt += 1 if t in unigram_list[fileid] else 0
		nt = sum(t in unigram_list[fileid] for fileid in train_set) # count #docs in which t occurs
		idf[t] = math.log(D / (1 + float(nt))) # 1 is added in the denominator for consistency during testing where nt may be = 0
	for i, fileid in enumerate(train_set):
		Ni = len(train_set[fileid]) # total #tokens in the document
		for j, t in enumerate(V):

			tf = unigram_list[fileid][t] if t in unigram_list[fileid] else 0.0
			# tf = (tf + alpha) / float(alpha*len(V) + Ni)
			tf = tf / float(Ni)
			X[i, j] = tf * idf[t]
	print('Training...')
	# clf1 = LinearSVC(multi_class='ovr')
	# clf1.fit(X, train_labels)

	clf1 = OneVsRestClassifier(LinearSVC(random_state=0))
	clf1.fit(X, train_labels)

	# create freq distribution of test data
	unigram_list_test = {}
	for fileid in test_set:
		tokens = test_set[fileid]
		unigram = create_unigram_dict(tokens)
		unigram_list_test[fileid] = unigram
	print('creating test data vector...')
	# create tf-idf feature vector for each document in test set
	x = np.zeros((len(test_set), len(V)))
	D_test = len(test_set)
	idf_test = {}
	for j, t in enumerate(V):
		nt = 0
		for fileid in test_set:
			nt += 1 if t in unigram_list_test[fileid] else 0
		# nt = sum(t in unigram_list_test[fileid] for fileid in test_set) # count #docs in which t occurs
		idf_test[t] = math.log(D_test / (1 + float(nt)))
	for i, fileid in enumerate(test_set):
		Ni = len(test_set[fileid]) # total #tokens in the document
		for j, t in enumerate(V):
			
			tf = unigram_list_test[fileid][t] if t in unigram_list_test[fileid] else 0.0
			# tf = (tf + alpha) / float(alpha*len(V) + Ni)
			tf = tf / float(Ni)
			x[i, j] = tf * idf_test[t]

	y = clf1.predict(x)
	# y2 = clf2.predict(x)

	accuracy1 = clf1.score(x, test_labels)
	# accuracy2 = clf2.score(x, test_labels)

	# print('Accuracy LinearSVC: ', accuracy1)
	print('Accuracy OneVsRestClassifier: ', accuracy1)

classification()