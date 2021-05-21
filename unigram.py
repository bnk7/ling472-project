# !/usr/bin/python3

import pandas as pd
import re
from math import log2

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
                self.df.loc[word, "cnt"] += 1
            else:
                row = pd.Series(data={"cnt": 1}, name=word)
                self.df = self.df.append(row, ignore_index=False)
        f.close()

    def train_unk(self): # Arshana
        num_unk = self.df.loc[self.df['cnt'] == 1].size
        # remove UNKed words
        self.df.drop(self.df[self.df['cnt'] == 1].index, inplace=True)
        # create UNK row
        row = pd.Series(data={'cnt': num_unk}, name="<UNK>")
        # append row
        self.df = self.df.append(row, ignore_index=False)

    # returns the smoothed probability of a single word
    def get_train_prob(self, cnt): # Brynna
        # P = (count of unigram + 1)/(sum of all counts + size of vocab)
        num_tokens = self.df.cnt.sum()
        num_types = self.df.shape[0]
        denominator = num_tokens + num_types
        prob = log2((cnt + 1)/denominator)
        return prob

    # applies Laplace smoothing and adds a "probability" column
    def train_prob(self): # Brynna
        self.df['MLE'] = self.df.cnt.apply(self.get_train_prob)

    def print_ngram(self): #Arshana
        # fix alphabetical
        self.df.index.name = "index"
        self.df.sort_values(by=['MLE', "index"], ascending = [False, True], inplace = True)

        for index, row in self.df.iterrows():
            print(index, round(row['MLE'], 3))

    def train(self, train_corpus):
        self.read_data(train_corpus)
        self.train_unk()
        self.train_prob()
        self.print_ngram()

    def score_unk(self, sent): # Anna
        # I have not tested any of this code yet.
        unked_sent = ""
        for word in sent.split():
            if word in self.df.index:
                unked_sent += word + " "
            else:
                unked_sent += "<UNK> "
        return unked_sent


    def score_prob(self, sent): # Brynna
        pass

    def calc_perplex(self, sum, count): # Arshana
        #untested
        # -1 for UNK token
        H = (float)sum / (len(self.df.index) - 1) * -1
        return round(2 ** H, 3)

    def score(self, test_corpus): # Anna
        """
        Reads in the test corpus and prints each line with its logged probability
        Prints out the perplexity of the system at the end. All rounded to the
        third decimal place.
        """
        # I have not tested any of this code yet.
        num_sent = 0
        total_prob = 0
        # read in data
        f = open(test_corpus, 'r')
        pattern = r'[^a-zA-Z0-9\s]'
        lines = f.readlines()
        f.close()
        # per sentence w/ prob
        for line in lines:
            num_sent += 1
            clean_line = re.sub(pattern=pattern, repl='', string=line)
            unked_line = self.score_unk(clean_line)
            prob = self.score_prob(unked_line)
            total_prob += prob
            print(line + " " + str(prob))
        # System's perplexity
        perplex = self.calc_perplex(total_prob, num_sent)
        print("Perplexity = " + str(perplex))

        # total_prob = 0
        # num_sent = 0
        # break test_corpus -> entire_file
            # num_sent++
            # print (line1)
            # score_unk (line1) -> return unked sent
                # add start and stop tokens
            # score_prob(sent) -> return prob1
            # total_prob += prob1
        # calc_perplex(total_prob, num_sent)

