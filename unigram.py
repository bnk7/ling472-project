# !/usr/bin/python3

import pandas as pd
import re

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
        self.df = pd.DataFrame(columns=["cnt"])

    def read_data(self, corpus): # Anna
        """
        reads in corpus data to fill in the unigram df with the counts for each word
        index: word
        cnt: count of word
        """
        f = open(corpus, 'r')
        pattern = r'[^a-zA-Z0-9\s]'
        entire_file = f.read()
        entire_file = re.sub(pattern=pattern, repl='', string=entire_file)
        entire_file = entire_file.split()
        for word in entire_file:
            if word in self.df.index:
                self.df.loc[word, "count"] += 1
            else:
                row = pd.Series(data={"count": 1}, name=word)
                self.df = self.df.append(row, ignore_index=False)
        print(self.df)
        f.close()


    def train_unk(self): # Arshana
	# num_unk = 0
	# loop through df
		# if an ngram appears only once then num_unk++
		# remove the row from df
	# add row to df with "<UNK>" as index and num_unk as value
        pass

    # returns the smoothed probability of a single word
    def get_train_prob(self, cnt):
        # P = (count of unigram + 1)/(sum of all counts + size of vocab)
        num_tokens = self.df.cnt.sum()
        num_types = self.df.shape[0]
        denominator = num_tokens + num_types
        prob = (cnt + 1)/denominator
        return prob

    # applies Laplace smoothing and adds a "probability" column
    def train_prob(self): # Brynna
        self.df['probability'] = self.df.apply(get_train_prob)

    def print_ngram(self): #Arshana
	# loop through df
		# output: "<index> <logged MLE rounded to 3rd decimal place>"
        pass

    def train(self, train_corpus):
        self.read_data(train_corpus)
        print('I am an unimplemented UNIGRAM train() method.')  # delete this!
        # train_prob()

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
