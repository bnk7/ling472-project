# !/usr/bin/python3

"""
read in data: one function
    split whitespace, or regex with no punctuation
    store in pandas, dataframe or series
    word as index

<UNK>: when word appears once


Smoothing to calculate the probabilities, add column to dataframe


print n-gram, MLE, add column to dataframe

score:

store the sum of probabilities for perplexity, but print probabilities as we go along

<unk> unseen words

probability for each sentence

Perplexity, sum each log probability for each sentence

"""



# TODO: Implement a Laplace-smoothed unigram model :)
class LanguageModel:

    def __init__(self):
        # df
        pass

    def read_data(self, corpus): # Anna
        pass

    def train_unk(self): # Arshana
	# num_unk = 0
	# loop through df
		# if an ngram appears only once then num_unk++
		# remove the row from df
	# add row to df with "<UNK>" as index and num_unk as value
        pass

    def train_prob(self): # Brynna
        pass

    def calculate_MLE(self): # Anna
        pass

    def print_ngram(self): #Arshana
	# loop through df
		# output: "<index> <logged MLE rounded to 3rd decimal place>"
        pass

    def train(self, train_corpus):
        print('I am an unimplemented UNIGRAM train() method.')  # delete this!

    def score_unk(self, sent):
        pass

    def score_prob(self, sent):
        pass

    def calc_perplex(self, sum, count):
        pass

    def score(self, test_corpus):
        # move through sentences, pass one at a time
        # use read_data
        # sum probability variable
        # count sentences variable
        print('I am an unimplemented UNIGRAM score() method.')  # delete this!
