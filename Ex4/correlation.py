import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
# text normalization by replacing all versions of you with 'you'
you_v = ["you", "your", "yours", "yourself", "yourselves", "you'd", "you'll", "you're", "you've"] # you related stop words suggested in https://www.link-assistant.com/seo-stop-words.html
normalized_tokens = [token if token.lower().strip("_-") not in you_v else 'you' for token in emma] # strip the token of any special characters. Text has instances of '__your__' which can be removed this way 

# form the normalized test and save it in a file
normalized_text = (' ').join(normalized_tokens)
f = open('emma_stemmed.txt', 'w')
f.write(normalized_text)
f.close()

# find correlation
limit = 50
word = 'you'

# read text and remove punctuations
text = open('emma_stemmed.txt').read().decode('utf8')
# create a word tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# tokenize the text into words 
tokens = tokenizer.tokenize(text)
# normalize to lowercase
tokens = [x.lower() for x in tokens]

p_w = tokens.count(word)
# print(p_w)
corr = []
fp = open('example.txt','w')
for d in range(1, limit+1):
	pd_ww = 0.0
	temp_text = tokens
	temp_text_length = len(temp_text)
	while temp_text_length > d:
		try: 
			idx = temp_text.index(word)
		except:
			break
		if idx+d < temp_text_length and temp_text[idx+d] == word:

			fp.write((' ').join(temp_text[idx: idx+d+1]))
			fp.write('\n')

			pd_ww += 1
		temp_text = temp_text[idx+1:temp_text_length]
	corr.append(len(tokens) * pd_ww /float(p_w * p_w))
	fp.write('\n')

fp.close()

plt.plot(np.arange(limit)+1, corr, marker="o")
plt.axhline(y=1, color='g', linestyle='-')
plt.title('Correlation Function for "you" ')
plt.xlabel('Distance between two occurences')
plt.ylabel('Correlation')
#save the plot in png format
plt.savefig('correlation.png')
plt.show()