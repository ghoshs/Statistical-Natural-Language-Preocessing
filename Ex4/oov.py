import os
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import operator

def preprocess(text):
	# create a word tokenizer
	tokenizer = RegexpTokenizer(r'\w+')
	# tokenize the text into words 
	tokens = tokenizer.tokenize(text)
	# convert text to lower case and ignore all unconvertable utf8 characters since these are likely punctuation marks
	tokens = [x.lower().encode('ascii','ignore') for x in tokens]
	return tokens

def create_vocab(text):
	tokens = preprocess(text)
	return len(tokens), dict(Counter(tokens))


def compute_oov(train_folder, test_file):
	train_list = [x for x in os.listdir(train_folder) if x.endswith('.txt')]

	vocab = {}
	vocab_size = {}
	vocab_test = {}
	size_test = 0.0
	oov = {}

	for filename in train_list:
		fp = open(train_folder + '/' + filename)
		text = fp.read().decode("utf-8")
		fp.close()
		_, vocab[text.split('.')[0]] = create_vocab(text)
		vocab_size[text.split('.')[0]] = len(vocab[text.split('.')[0]])

	fp = open(test_file)
	text = fp.read().decode("utf-8")
	fp.close()
	size_test, vocab_test = create_vocab(text)

	for i in vocab: # for each vocabulary
		unseen = 0.0
		for token in vocab_test:
			if token not in vocab[i]:
				unseen += vocab_test[token]

		oov[i] = (unseen / float(size_test)) * 100.0

	x_arranged = sorted(vocab_size.items(), key=operator.itemgetter(1))
	x = [size for idx, size in x_arranged]
	y = [oov[idx] for idx, size in x_arranged]
	plt.loglog(x, y, marker='o')
	plt.title('OOV Rate vs. Vocabulary Size')
	plt.xlabel('Log Vocabulary Size')
	plt.ylabel('Log OOV rate in %age')
	plt.grid(True)
	#save the plot in png format
	plt.savefig('oov.png')
	plt.show()

compute_oov('Materials_Ex4/train', 'Materials_Ex4/test/test.txt')