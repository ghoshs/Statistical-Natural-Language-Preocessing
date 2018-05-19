import os
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
from collections import Counter
import math
import operator
import numpy as np
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

ps = PorterStemmer()
# wl = WordNetLemmatizer()

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

def classification(trainpath, testfile, stopfile):
	tokens_list = [] # list of all tokens in each class represented by doclist
	unigram_list = [] # list of unigram frequency in each class
	classlist = [x for x in os.listdir(trainpath) if x.endswith('.txt')]
	N = 0 # total number of tokens in the training corpus
	# assuming class probablity is represented by document frequency which in this case is equal
	# P_c = (1/float(len(classlist)))
	V = {}

	# tokenize all documents in each class and extend the vocabulary
	for doc in classlist: 
		tokens = preprocess(doc, trainpath, stopfile)
		unigram = create_unigram_dict(tokens)
		tokens_list.append(tokens)
		unigram_list.append(unigram)
		for t in unigram:
			if t not in V:
				V[t] = unigram[t]
			else:
				V[t] += unigram[t]
		N += len(tokens)

	# ---------------------- Calculate TF-IDF -------------------------- #
	# calculate tfidf for every vocabulary word
	tfidf = {}
	idf = {}
	num_classes = len(classlist) # here # classes = # documents
	for t in V:
		nt = sum(t in unigram_list[i] for i in range(num_classes)) # total #docs which contain t^th feature 
		idf[t] = math.log(num_classes / (1 + float(nt)))
		# calculate tf as #occurences of t in entire train corpus / #tokens in training corpus
		# for document representation tf = #occurences of t in the document / #tokens in the document
		tf = V[t] / float(N)
		tfidf[t] = tf*idf[t]

	top_500 = sorted(tfidf.items(), key=operator.itemgetter(1), reverse=True)[0:500]
	fp = open('Top_500_tf-idf_features.txt', 'w')
	for i in top_500:
		fp.write(i[0] + " : " + str(i[1]) + '\n')
	fp.close()
	print('Top 500 TF-IDF features saved in: Top_500_tf-idf_features.txt \n')

	# -------------------------- CLASSIFICATION --------------------------- #

	# create tf-idf representation for each Document
	# alpha = 1.0
	X1 = np.zeros((num_classes, 500))
	Y1 = np.zeros((num_classes,), dtype=np.int)
	for i in range(num_classes):
		Ni = len(tokens_list[i])
		for idx, (t, _) in enumerate(top_500):
			tf = unigram_list[i][t] / float(Ni) if t in unigram_list[i] else 0.0
			# tf = (tf + alpha) / float(Ni + alpha*len(V))
			X1[i, idx] = tf * idf[t]
		Y1[i] = i

	# ----------------- TRAINING NAIVE BAYES------------------ # 
	clf = MultinomialNB()
	clf.fit(X1, Y1)

	# ----------------- TRAINING KNN ------------------ # 
	knn2 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
	knn2.fit(X1, Y1)

	# preprocess test data
	x1 = np.zeros((1, 500))
	tokens = preprocess(testfile, '.', stopfile)
	unigram = create_unigram_dict(tokens)
	Ni = len(tokens)


	for idx, (t, _) in enumerate(top_500):
		# nt = sum(t in unigram_list[i] for i in range(num_classes))# total #docs which contain t^th feature 
		# nt  = nt +1 if t in unigram else 0
		# idf = math.log((num_classes+1) / (1 + float(nt)))
		tf = unigram[t]/ float(Ni) if t in unigram else 0.0
		# tf = (tf + alpha) / float(Ni + alpha*len(V))
		idf = math.log(0.5) if t in unigram else 0.0 
		x1[0, idx] = tf * idf

	# ----------------- TESTING NB------------------- #
	predict = clf.predict(x1)
	catgry = classlist[predict[0]].split('.')[0]
	print('NB Classifier with top 500 TF-IDF features: Test file is classified as : ' + catgry + '\n')

	# ----------------- TESTING KNN------------------- #
	predict = knn2.predict(x1)
	catgry = classlist[predict[0]].split('.')[0]
	print('KNN Classifier with top 500 TF-IDF features: Test file is classified as : ' + catgry + '\n')

	# --------------- Using TF ------------------#
	
	# create feature vector for each document
	X2 = np.zeros((num_classes, len(V)))
	Y2 = np.zeros((num_classes,), dtype=np.int) 
	for i in range(num_classes):
		Ni = len(tokens_list[i])
		for idx, (t, _) in enumerate(top_500):
			tf = unigram_list[i][t] / float(Ni) if t in unigram_list[i] else 0.0
			# tf = (tf + alpha) / float(Ni + alpha*len(V))
			X2[i, idx] = tf
		Y2[i] = i

	# ----------------- TRAINING ------------------ # 
	knn1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
	knn1.fit(X2, Y2)

	# ----------------- TESTING ------------------- #
	# preprocess test data
	x2 = np.zeros((1, len(V)))
	tokens = preprocess(testfile, '.', stopfile)
	unigram = create_unigram_dict(tokens)
	Ni = len(tokens)
	for idx, (t, _) in enumerate(top_500):
		tf = unigram[t] / float(Ni) if t in unigram else 0.0
		# tf = (tf + alpha) / float(Ni + alpha*len(V))
		x2[0, idx] = tf

	predict = knn1.predict(x2)
	catgry = classlist[predict[0]].split('.')[0]
	print('KNN Classifier with TF features: Test file is classified as : ' + catgry + '\n')

classification('./Materials_Ex7/train', 'Materials_Ex7/test_2.txt', './Materials_Ex7/stopwords.txt')