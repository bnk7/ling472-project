# !/usr/bin/python3

from math import log2
import pandas as pd

# TODO: Implement a Laplace-smoothed bigram model :)
class LanguageModel:

    def __init__(self):
        self.unigram_df = pd.DataFrame()
        self.bigram_df = pd.DataFrame()

    def read_data(self, corpus): # Brynna
        sentences = corpus.readlines()
        for line in sentences:
            line = line.strip()
            line = "<s> " + line + " </s>"
        # deal with double dashes?
        # adapt unigram code to make unigram df
        # make bigram df with bigram as index and each individual word as a column

    def train_unk(self): # Anna
        pass

    def smoothing(self): # Arshana
	# for each w1 in vocab list
		# for each w2 in vocab list
			# if "<w1> <w2>" is not an index in df then add it to df with 0 occurrences
	# loop through df
		# add 1 to each ngram in df
        pass

    def probability(self): # Anna
        pass

    def calculate_MLE(self): # Arshana
	# loop through df
		# add a column with count / count(w2)
        pass

    # prints bigrams and their logged MLEs, rounded to 3 decimal places
    # and sorted by logged MLE (descending) and then bigram (alphabetical)
    def print_ngram(self): # Brynna
        # add loggedMLE column
        self.bigram_df['loggedMLE'] = self.bigram_df.apply(log2)
        # sort alphabetically
        self.bigram_df.sort_index(inplace = True)
        # sort by loggedMLE
        self.bigram_df.sort_values(['loggedMLE'], ascending = False, inplace = True)
        # print
        print(round(self.bigram_df.loggedMLE, 3))

    def train(self, train_corpus):
        print('I am an unimplemented BIGRAM train() method.')  # delete this!
        # read_data(train_corpus)

    def score(self, test_corpus):
        print('I am an unimplemented BIGRAM score() method.')  # delete this!
