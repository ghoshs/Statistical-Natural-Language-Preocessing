import nltk
from nltk.corpus import brown
import operator
import numpy as np
import matplotlib.pyplot as plt
import math

# get a list of all tokens in the Brown corpus in order of their occurrence in text 
tokens = brown.words()

# convert the unicode entries to string
string_tokens = [str(x) for x in tokens]

#convert the tokens to lower case
lowercase_tokens = [x.lower() for x in string_tokens]

# calulate frequencies of 'in' and 'the' and use them to find their relative frequencies
freq_in = lowercase_tokens.count('in')
freq_the = lowercase_tokens.count('the')

total_tokens =  len(lowercase_tokens)

relative_freq_in = freq_in / float(total_tokens) 
relative_freq_the = freq_the / float(total_tokens)

print('Relative Frequency "in ": '+str(relative_freq_in))
print('Relative Frequency "the": '+str(relative_freq_the))

total_bigrams = total_tokens

# create a dict of bigrams with first word = 'in'
joint_frequency_in = {}

for idx, token in enumerate(lowercase_tokens):
	if token == 'in' and idx < len(lowercase_tokens)-1:
		next_word = lowercase_tokens[idx+1]
		bigram = next_word
		if bigram not in joint_frequency_in:
			joint_frequency_in[bigram] = 1;
		else:
			joint_frequency_in[bigram] = joint_frequency_in[bigram] + 1

# create a dict of bigrams with first word = 'in'
joint_frequency_the = {}

for idx, token in enumerate(lowercase_tokens):
	if token == 'the' and idx < len(lowercase_tokens)-1:
		next_word = lowercase_tokens[idx+1]
		bigram = next_word
		if bigram not in joint_frequency_in:
			joint_frequency_the[bigram] = 1
		else:
			joint_frequency_the[bigram] = joint_frequency_in[bigram] + 1

# calculate the absolute frequencies of the bigrams
joint_probability_in = {token: joint_frequency_in[token]/ float(total_bigrams) for token in joint_frequency_in}
joint_probability_the = {token: joint_frequency_the[token]/ float(total_bigrams) for token in joint_frequency_the}

# conditional probability for words staring with in
conditional_in = {x: joint_probability_in[x] / float(relative_freq_in) for x in joint_probability_in}

# conditional probability for words staring with the
conditional_the = {x: joint_probability_the[x] / float(relative_freq_the) for x in joint_probability_the}

# 20 most frequent tokens starting with in 
top_twenty_in = sorted(conditional_in.items(), key=operator.itemgetter(1), reverse=True)[0:20]

# 20 most frequent tokens starting with the
top_twenty_the = sorted(conditional_the.items(), key=operator.itemgetter(1), reverse=True)[0:20]

# Normalized Frequency distribution of the top 20 words
prob_dist1 = [lowercase_tokens.count(token) / float(total_tokens) for token, _ in top_twenty_in]
prob_dist2 = [lowercase_tokens.count(token) / float(total_tokens) for token, _ in top_twenty_the]

#print
print('*****************************************************')
print("Top 20 words for 'in' :")
for i, (token, _) in enumerate(top_twenty_in):
	print(str(i+1)+'. '+token)

print("Top 20 words for 'the' :")
for i, (token, _) in enumerate(top_twenty_the):
	print(str(i+1)+'. '+token)

# plot for in and the
print('*****************************************************')
print('shape of x', np.arange(20).shape)
print('shape of y1', len(prob_dist1))
print('shape of y2', len(prob_dist2))
print('*****************************************************')

indist, = plt.plot(np.arange(20), prob_dist1, '-ro', label = "in", linewidth = 2.0)
plt.xlabel('Word Rank')
plt.ylabel('Probability Distribution')
plt.legend(handles = [indist], numpoints = 1)
plt.savefig('brown_in.png')
plt.figure()
thedist, = plt.plot(np.arange(20), prob_dist2, '-bo', label = "the", linewidth = 2.0)
plt.xlabel('Word Rank')
plt.ylabel('Probability Distribution')
plt.legend(handles = [thedist], numpoints = 1)
plt.savefig('brown_the.png')

Exp_in = 0.0
for token in conditional_in:
	px = conditional_in[token]
	Exp_in = Exp_in - px*math.log(px, 2)


Exp_the = 0.0
for token in conditional_the:
	px = conditional_the[token]
	Exp_the = Exp_the - px*math.log(px, 2)

print('"in" Expectation :' + str(Exp_in))
print('"the" Expectation :' + str(Exp_the))
print('*****************************************************')

plt.show()