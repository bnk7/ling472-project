# !/usr/bin/python3

from math import log2
import pandas as pd
import re

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
        """
        Unks the unigram df. Using the unked word list from the unigram,
        also unks the bigram df.
        """
        # UNIGRAM
        is_one = self.unigram_df["cnt"] == 1
        # count of words to unk
        unk_count = self.unigram_df[is_one].size
        # all unked words
        unked_words = self.unigram_df[is_one].index
        # dataframe with none of the unked words
        unked_df = self.unigram_df[self.unigram_df["cnt"] != 1]
        # create row for unk
        row = pd.Series(data={"cnt": unk_count}, name="<UNK>")
        # append row and save to class variable
        self.unigram_df = unked_df.append(row, ignore_index=False)
        
        # BIGRAM
        # replace words that need unk from unigram unk list
        unked_bigram = self.bigram_df.replace(unked_words, "<UNK>")
        # sum counts with same w1, w2
        unked_bigram = unked_bigram.groupby(['w1', 'w2']).sum()
        # new dataframe to save groupby with correct indexes and column names
        unked_bigram2 = pd.DataFrame()
        for tup in unked_bigram.index:
            w1, w2 = tup
            row = pd.Series(data=[w1, w2, unked_bigram.loc[w1, w2]["cnt"]], name=w1 + " " + w2)
            unked_bigram2 = unked_bigram2.append(row, ignore_index=False)
        unked_bigram2.columns = ["w1", "w2", "cnt"]
        # save to class variable
        self.bigram_df = unked_bigram2
        

    def smoothing(self): # Arshana
	# for each w1 in vocab list
		# for each w2 in vocab list
			# if "<w1> <w2>" is not an index in df then add it to df with 0 occurrences
	# loop through df
		# add 1 to each ngram in df
        pass

    def train_prob(self): # Anna
        """
        Adds a probability column to the bigram df.
        """
        # for each row in bigram df
        for index, row in self.bigram_df.iterrows():
            # bigram count
            num = row['cnt']
            # w1 count + vocab count (no <s>)
            denom = self.unigram_df.loc[row['w1'], 'cnt'] + self.unigram_df.size - 1
            # prob: count / w1 count + vocab count
            # add column to bigram df
            self.bigram_df.loc[index, 'prob'] = float(num)/denom

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
