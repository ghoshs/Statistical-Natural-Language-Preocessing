# import nltk
import numpy as np
import matplotlib.pyplot as plt
import operator
import math

# function to read a giveb file and return a sorted list of words and their frequencies
def read_file(filename):
	word_dict = {}
	fp = open(filename, 'rb')
	print('Reading ' + filename)
	
	# count the frequencies of each unique word in the document
	for sent in fp:
		for word in sent.split():
			if word not in word_dict:
				
				# create an entry for the word encountered for the first time in the document
				word_dict[word] = 1
			else:
				
				# increase word count on each occurrence
				word_dict[word] += 1
	print('Dictionary formed! Words: ' +str(len(word_dict)))
	return sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
if __name__ == '__main__':
	file1 = 'ACCEnglish.txt'
	file2 = 'ACCGerman.txt'
	file3 = 'ACCFrench.txt'
	
	# call function for creating word dictionaries of the text documents
	dict_english = read_file(file1)
	dict_french = read_file(file2)
	dict_german = read_file(file3)

	# arange the plot points in log scale
	x1 = [math.log(x+1) for x in np.arange(len(dict_english))]
	y1 = [math.log(x) for i, x in dict_english]
	x2 = [math.log(x+1) for x in np.arange(len(dict_french))]
	y2 = [math.log(x) for i, x in dict_french]
	x3 = [math.log(x+1) for x in np.arange(len(dict_german))]
	y3 = [math.log(x) for i, x in dict_german]

	# create plot handles for English, French and German text
	english, = plt.plot(x1, y1, 'r-', label="English")
	french, = plt.plot(x2, y2, 'b-', label="French")
	german, = plt.plot(x3, y3, 'g-', label="German")
	plt.xlabel('Log Word Rank')
	plt.ylabel('Log Frequencies')
	plt.legend(handles= [english, french, german])

	#save the plot in png format
	plt.savefig('Zipf.png')
	plt.show()