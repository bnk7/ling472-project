from math import log2
import pandas as pd
import re
# !/usr/bin/python3


# TODO: Implement a Laplace-smoothed trigram model :)
class LanguageModel:

    def __init__(self):
        self.unigram = pd.DataFrame(columns=["cnt"])
        self.bigram = pd.DataFrame(columns=["w1", "w2", "cnt"])
        self.trigram = pd.DataFrame(columns=["w1", "w2", "w3", "cnt"])

    def read_data(self, corpus): # Arshana
        # start and stop tokens
            # two start token on the first sentence
            # two stop on the last
            # rest one
            # sentence on each line
            # or double dashes
        # three dataframes: unigram index, bigram as index and each individual word as a column
        # trigram index and three columns for words
	
	# string split on end-of-sentence char(s)
	# loop through sentences
		# insert start/stop tokens
		# split sentences and combine all split sentences into one list
	# loop through list
		# for df
			# create trigram indices
			# add w1, w2, w3 columns
			# add count column and increment as appropriate
		# for vocab df
			# create unigram indices
			# add count column and increment as appropriate
        pass

    def train_unk(self): # Brynna
        pass

    def smoothing(self): # Anna
        pass

    def train_prob(self): # Arshana
        pass

    def print_ngram(self): # Anna
        """
        log2 the probability and save to new MLE column
        prints out each trigram with its logged MLE
        """
        # <w1> <w2> <w3> <log2(P(w3|w1 w2))>
        # highest to lowest prob, 3 decimal place rounded
        # then alphabetical
        self.trigram["loggedMLE"] = self.trigram["prob"].apply(log2)
        self.trigram.sort_index(inplace=True)
        self.trigram.sort_values(by=['prob'], inplace=True, ascending=False)
        print(round(self.trigram["loggedMLE"], 3))


    def train(self, train_corpus):
        print('I am an unimplemented TRIGRAM train() method.')  # delete this!

    def score(self, test_corpus):
        print('I am an unimplemented TRIGRAM score() method.')  # delete this!
